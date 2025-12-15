"""
Language Model Wrapper
======================

Wrapper for quantized language models using llama-cpp-python.
Supports GGUF format models for efficient local inference.

Why Quantization?
-----------------
Quantization reduces model precision (e.g., from FP32 to INT8 or INT4),
which dramatically reduces memory usage and improves inference speed
with minimal quality degradation. For example:
- Llama 2 7B FP16: ~14GB VRAM
- Llama 2 7B Q4_K_M: ~4GB RAM (works on 16GB laptop)

We use GGUF format with llama.cpp for several reasons:
1. Efficient CPU inference with SIMD optimizations
2. Optional GPU acceleration (CUDA, Metal, OpenCL)
3. Small memory footprint
4. Good quality retention at 4-bit quantization
"""

import os
import time
from typing import Optional, List, Generator, Dict, Any
from pathlib import Path

from .config import LLMConfig


class LocalLLM:
    """
    Wrapper for local quantized language models.
    
    Uses llama-cpp-python for efficient inference on commodity hardware.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the language model.
        
        Args:
            config: LLM configuration object
        """
        self.config = config
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the quantized model into memory."""
        from llama_cpp import Llama
        
        if not self.config.model_path:
            raise ValueError(
                "Model path not specified. Please download a GGUF model and "
                "set the model_path in config. Recommended: "
                "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
            )
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please download a GGUF model file."
            )
        

        n_threads = self.config.n_threads
        if n_threads == 0:
            n_threads = os.cpu_count() or 4
            # Leave some threads for the system
            n_threads = max(1, n_threads - 2)
        
        print(f"Loading model from {model_path}...")
        print(f"Using {n_threads} CPU threads")
        
        start_time = time.time()
        
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=self.config.n_ctx,
            n_threads=n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False,
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Generate text from the model.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system instructions
            max_tokens: Override max tokens from config
            temperature: Override temperature from config
            stream: If True, yield tokens as they're generated
            
        Returns:
            Generated text or generator of tokens if streaming
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        

        full_prompt = self._build_prompt(prompt, system_prompt)
        

        gen_params = {
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
            "top_p": self.config.top_p,
            "repeat_penalty": self.config.repeat_penalty,
            "stop": self.config.stop_sequences,
        }
        
        if stream:
            return self._generate_stream(full_prompt, gen_params)
        else:
            return self._generate_complete(full_prompt, gen_params)
    
    def _build_prompt(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Build a properly formatted prompt for the model.
        
        Uses a chat template format compatible with most instruction-tuned models.
        Note: We don't add <s> here as llama.cpp adds it automatically.
        """
        # Mistral/Llama chat format (without leading <s> to avoid duplication)
        if system_prompt:
            return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        else:
            return f"[INST] {user_prompt} [/INST]"
    
    def _generate_complete(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate complete response (non-streaming)."""

        prompt = self._truncate_prompt(prompt, params["max_tokens"])
        
        response = self.model(
            prompt,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            repeat_penalty=params["repeat_penalty"],
            stop=params["stop"],
            echo=False,
        )
        
        return response["choices"][0]["text"].strip()
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Truncate prompt if it exceeds context window.
        
        Preserves the instruction format by truncating context in the middle.
        """
        # Reserve tokens for generation
        max_prompt_tokens = self.config.n_ctx - max_tokens - 100  # 100 token buffer
        
        prompt_tokens = self.count_tokens(prompt)
        
        if prompt_tokens <= max_prompt_tokens:
            return prompt
        
        # Need to truncate - find the context section and trim it
        # Look for common context markers
        context_start_markers = ["Context:", "CONTEXT:", "Based on the following"]
        context_end_markers = ["Question:", "QUESTION:", "[/INST]"]
        
        context_start = -1
        context_end = len(prompt)
        
        for marker in context_start_markers:
            idx = prompt.find(marker)
            if idx != -1:
                context_start = idx + len(marker)
                break
        
        for marker in context_end_markers:
            idx = prompt.rfind(marker)
            if idx != -1 and idx > context_start:
                context_end = idx
                break
        
        if context_start == -1:
            # No context section found, truncate from the end
            # Binary search for right length
            low, high = 0, len(prompt)
            while low < high:
                mid = (low + high + 1) // 2
                if self.count_tokens(prompt[:mid]) <= max_prompt_tokens:
                    low = mid
                else:
                    high = mid - 1
            return prompt[:low] + "\n[/INST]"
        
        # Truncate the context section from the middle
        context = prompt[context_start:context_end]
        prefix = prompt[:context_start]
        suffix = prompt[context_end:]
        
        # Calculate how much context we can keep
        fixed_tokens = self.count_tokens(prefix + suffix + "\n\n[TRUNCATED]\n\n")
        available_for_context = max_prompt_tokens - fixed_tokens
        
        if available_for_context <= 0:
            # Very limited space, just keep minimal context
            return prefix + "\n[Context truncated due to length]\n" + suffix
        
        # Keep beginning and end of context, truncate middle
        context_tokens = self.count_tokens(context)
        if context_tokens <= available_for_context:
            return prompt
        
        # Split context and keep proportional parts from start and end
        half_available = available_for_context // 2
        
        # Find truncation points
        lines = context.split('\n')
        start_lines = []
        end_lines = []
        start_tokens = 0
        end_tokens = 0
        
        # Take lines from start
        for line in lines:
            line_tokens = self.count_tokens(line + '\n')
            if start_tokens + line_tokens <= half_available:
                start_lines.append(line)
                start_tokens += line_tokens
            else:
                break
        
        # Take lines from end
        for line in reversed(lines):
            line_tokens = self.count_tokens(line + '\n')
            if end_tokens + line_tokens <= half_available:
                end_lines.insert(0, line)
                end_tokens += line_tokens
            else:
                break
        
        truncated_context = '\n'.join(start_lines) + '\n\n[...TRUNCATED...]\n\n' + '\n'.join(end_lines)
        
        return prefix + truncated_context + suffix
    
    def _generate_stream(
        self, prompt: str, params: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Generate response with streaming."""

        prompt = self._truncate_prompt(prompt, params["max_tokens"])
        
        stream = self.model(
            prompt,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            repeat_penalty=params["repeat_penalty"],
            stop=params["stop"],
            echo=False,
            stream=True,
        )
        
        for chunk in stream:
            token = chunk["choices"][0]["text"]
            yield token
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return len(self.model.tokenize(text.encode()))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.config.model_path,
            "context_length": self.config.n_ctx,
            "n_threads": self.config.n_threads or os.cpu_count(),
            "n_gpu_layers": self.config.n_gpu_layers,
        }


def create_llm(config: LLMConfig) -> LocalLLM:
    """
    Factory function to create an LLM instance.
    
    Args:
        config: LLM configuration
        
    Returns:
        LLM instance
    """
    return LocalLLM(config)

