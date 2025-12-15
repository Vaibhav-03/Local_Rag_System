"""
Content Guardrails
==================

Implements safety constraints and topic filtering for the RAG system.

Guardrails serve multiple purposes:
1. Prevent harmful or inappropriate content generation
2. Keep the system focused on intended use cases
3. Protect against prompt injection attacks
4. Ensure compliance with usage policies

Implementation Strategy:
- Keyword-based blocking for explicit harmful topics
- Query validation for length and format
- Optional domain restriction to specific topics
- Prompt injection detection
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .config import GuardrailsConfig


class GuardrailAction(Enum):
    """Actions that can be taken by guardrails."""
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    MODIFY = "modify"


@dataclass
class GuardrailResult:
    """Result of guardrail evaluation."""
    action: GuardrailAction
    reason: str
    modified_query: Optional[str] = None
    
    @property
    def is_allowed(self) -> bool:
        return self.action in (GuardrailAction.ALLOW, GuardrailAction.WARN, GuardrailAction.MODIFY)


class ContentGuardrails:
    """
    Implements content safety guardrails for queries and responses.
    """
    
    def __init__(self, config: GuardrailsConfig):
        """
        Initialize guardrails.
        
        Args:
            config: Guardrails configuration
        """
        self.config = config
        

        self._blocked_patterns = [
            re.compile(rf'\b{re.escape(topic)}\b', re.IGNORECASE)
            for topic in config.blocked_topics
        ]
        

        self._injection_patterns = [
            re.compile(r'ignore\s+(previous|above|all)\s+(instructions?|prompts?)', re.IGNORECASE),
            re.compile(r'disregard\s+(previous|above|all)', re.IGNORECASE),
            re.compile(r'forget\s+(everything|your\s+instructions)', re.IGNORECASE),
            re.compile(r'you\s+are\s+now\s+(a|an|in)', re.IGNORECASE),
            re.compile(r'pretend\s+(to\s+be|you\s+are)', re.IGNORECASE),
            re.compile(r'system\s*:\s*', re.IGNORECASE),
            re.compile(r'\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>', re.IGNORECASE),
        ]
    
    def check_query(self, query: str) -> GuardrailResult:
        """
        Check if a query is allowed.
        
        Args:
            query: User query to check
            
        Returns:
            GuardrailResult with action and reason
        """
        if not self.config.enabled:
            return GuardrailResult(GuardrailAction.ALLOW, "Guardrails disabled")
        

        if len(query) > self.config.max_query_length:
            return GuardrailResult(
                GuardrailAction.BLOCK,
                f"Query too long ({len(query)} chars). Maximum: {self.config.max_query_length}"
            )
        

        if not query.strip():
            return GuardrailResult(
                GuardrailAction.BLOCK,
                "Empty query"
            )
        

        for pattern in self._injection_patterns:
            if pattern.search(query):
                return GuardrailResult(
                    GuardrailAction.BLOCK,
                    "Query appears to contain prompt injection attempt"
                )
        

        for i, pattern in enumerate(self._blocked_patterns):
            if pattern.search(query):
                topic = self.config.blocked_topics[i]
                return GuardrailResult(
                    GuardrailAction.BLOCK,
                    f"Query contains blocked topic: {topic}"
                )
        

        if self.config.allowed_domains:
            domain_match = self._check_domain(query)
            if not domain_match:
                return GuardrailResult(
                    GuardrailAction.WARN,
                    "Query may be outside allowed domains. Proceeding with caution."
                )
        
        return GuardrailResult(GuardrailAction.ALLOW, "Query passed all checks")
    
    def _check_domain(self, query: str) -> bool:
        """Check if query falls within allowed domains."""
        query_lower = query.lower()
        return any(
            domain.lower() in query_lower
            for domain in self.config.allowed_domains
        )
    
    def check_response(self, response: str) -> GuardrailResult:
        """
        Check if a generated response is appropriate.
        
        Args:
            response: Generated response to check
            
        Returns:
            GuardrailResult with action and reason
        """
        if not self.config.enabled:
            return GuardrailResult(GuardrailAction.ALLOW, "Guardrails disabled")
        

        for i, pattern in enumerate(self._blocked_patterns):
            if pattern.search(response):
                topic = self.config.blocked_topics[i]
                return GuardrailResult(
                    GuardrailAction.BLOCK,
                    f"Response contains blocked content: {topic}"
                )
        
        return GuardrailResult(GuardrailAction.ALLOW, "Response passed all checks")
    
    def get_rejection_message(self) -> str:
        """Get the rejection message for blocked queries."""
        return self.config.rejection_message
    
    def sanitize_query(self, query: str) -> str:
        """
        Sanitize a query by removing potentially harmful elements.
        
        Args:
            query: Original query
            
        Returns:
            Sanitized query
        """
        sanitized = query
        

        for pattern in self._injection_patterns:
            sanitized = pattern.sub('', sanitized)
        

        sanitized = ' '.join(sanitized.split())
        
        return sanitized.strip()


class QueryRefiner:
    """
    Helps refine ambiguous or unclear queries.
    
    Implements prompt engineering techniques to improve query quality
    and reduce uncertainty.
    """
    
    def __init__(self):

        self._ambiguous_patterns = [
            (re.compile(r'^(what|how|why|when|where|who)\s*\??$', re.IGNORECASE), 
             "Your question seems incomplete. Could you provide more details?"),
            (re.compile(r'^(it|this|that|they)\s', re.IGNORECASE),
             "Could you clarify what you're referring to?"),
            (re.compile(r'^(help|please|thanks?)$', re.IGNORECASE),
             "I'd be happy to help! What would you like to know?"),
        ]
        

        self._clarification_keywords = [
            'maybe', 'perhaps', 'not sure', 'i think', 'possibly',
            'something about', 'anything about', 'stuff about'
        ]
    
    def analyze_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Analyze if a query needs refinement.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (needs_refinement, clarification_prompt)
        """
        query_lower = query.lower().strip()
        

        for pattern, prompt in self._ambiguous_patterns:
            if pattern.match(query):
                return True, prompt
        

        words = query.split()
        if len(words) < 3:
            return True, "Your query is quite brief. Could you provide more context or details?"
        

        if any(kw in query_lower for kw in self._clarification_keywords):
            return True, "It sounds like you're uncertain about your query. Let me help clarify - what specific aspect would you like to know about?"
        
        return False, None
    
    def suggest_related_queries(self, query: str, context: List[str]) -> List[str]:
        """
        Suggest related queries based on context.
        
        Args:
            query: Original query
            context: Retrieved context passages
            
        Returns:
            List of suggested follow-up queries
        """
        suggestions = []
        
        # Extract potential topics from context
        # This is a simple heuristic - could be enhanced with NLP
        all_text = ' '.join(context)
        
        # Simple suggestions based on query type
        if query.lower().startswith('what'):
            suggestions.append(f"Why {query[5:].strip()}?")
            suggestions.append(f"How does {query[5:].strip()} work?")
        elif query.lower().startswith('how'):
            suggestions.append(f"What are the alternatives to {query[4:].strip()}?")
        
        return suggestions[:3]  # Limit to 3 suggestions

