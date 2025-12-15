"""
Command Line Interface

Usage:
    python -m src.cli                    # Interactive mode
    python -m src.cli query "question"   # Single query
    python -m src.cli index              # Build index
"""

import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.live import Live

from .config import RAGConfig, get_default_config
from .rag import RAGPipeline, RAGResponse


console = Console()


def print_banner():
    """Print the application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                   🔍 LOCAL RAG SYSTEM 🔍                       ║
║         Retrieval-Augmented Generation on Your Laptop         ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_response(response: RAGResponse):
    """Print a formatted RAG response."""

    console.print()
    console.print(Panel(
        Markdown(response.answer),
        title="💬 Response",
        border_style="green",
    ))
    

    if response.sources:
        table = Table(title="📚 Sources", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Source", style="green")
        table.add_column("Relevance", style="yellow", width=12)
        table.add_column("Preview", style="dim")
        
        for result in response.sources:
            source_name = result.document.source or f"Document {result.document.doc_id}"
            relevance = f"{int(result.score * 100)}%"
            preview = result.document.text[:80] + "..." if len(result.document.text) > 80 else result.document.text
            preview = preview.replace("\n", " ")
            
            table.add_row(
                str(result.rank),
                source_name,
                relevance,
                preview,
            )
        
        console.print(table)
    

    console.print(
        f"\n⏱️  [dim]Retrieval: {response.retrieval_time*1000:.0f}ms | "
        f"Generation: {response.generation_time*1000:.0f}ms | "
        f"Total: {response.total_time*1000:.0f}ms | "
        f"Tokens: {response.tokens_used}[/dim]"
    )
    
   
    if response.guardrail_warning:
        console.print(f"\n⚠️  [yellow]{response.guardrail_warning}[/yellow]")


def stream_response(pipeline: RAGPipeline, query: str):
    """Stream a response with live output."""
    console.print()
    console.print(Panel.fit("💬 Response", border_style="green"))
    
    response_text = ""
    sources = []
    
    with Live(console=console, refresh_per_second=10) as live:
        generator = pipeline.query(query, stream=True)
        
        for token in generator:
            if isinstance(token, str):
                response_text += token
                live.update(Markdown(response_text))
            elif isinstance(token, RAGResponse):

                sources = token.sources
                break
    
    console.print()
    

    if sources:
        table = Table(title="📚 Sources", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Source", style="green")
        table.add_column("Relevance", style="yellow", width=12)
        
        for result in sources:
            source_name = result.document.source or f"Document {result.document.doc_id}"
            relevance = f"{int(result.score * 100)}%"
            table.add_row(str(result.rank), source_name, relevance)
        
        console.print(table)


@click.group(invoke_without_command=True)
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config YAML file')
@click.option('--model', '-m', type=click.Path(exists=True), help='Path to GGUF model file')
@click.option('--corpus', '-d', type=click.Path(exists=True), help='Path to corpus directory')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config, model, corpus, verbose):
    """
    Local RAG System - Run a retrieval-augmented LLM on your laptop.
    
    \b
    Examples:
        rag-cli                          # Start interactive mode
        rag-cli query "What is Python?"  # Single query
        rag-cli index                    # Build document index
        rag-cli stats                    # Show system stats
    """
    ctx.ensure_object(dict)
    
    
    if config:
        rag_config = RAGConfig.from_yaml(config)
    else:
        rag_config = get_default_config()
    

    if model:
        rag_config.llm.model_path = model
    if corpus:
        rag_config.corpus_dir = corpus
    if verbose:
        rag_config.verbose = True
    
    ctx.obj['config'] = rag_config
    
    
    if ctx.invoked_subcommand is None:
        interactive_mode(rag_config)


@cli.command()
@click.argument('question')
@click.option('--stream', '-s', is_flag=True, help='Stream the response')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def query(ctx, question, stream, output_json):
    """Ask a single question."""
    config = ctx.obj['config']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading RAG pipeline...", total=None)
        pipeline = RAGPipeline(config)
    
    if stream:
        stream_response(pipeline, question)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Thinking...", total=None)
            response = pipeline.query(question)
        
        if output_json:
            import json
            console.print_json(json.dumps(response.to_dict(), indent=2))
        else:
            print_response(response)


@cli.command()
@click.option('--corpus', '-d', type=click.Path(exists=True), help='Corpus directory')
@click.pass_context
def index(ctx, corpus):
    """Build or rebuild the document index."""
    config = ctx.obj['config']
    
    if corpus:
        config.corpus_dir = corpus
    
    console.print(f"\n📁 Indexing corpus from: [cyan]{config.corpus_dir}[/cyan]\n")
    
    corpus_path = Path(config.corpus_dir)
    if not corpus_path.exists():
        console.print(f"[red]Error: Corpus directory not found: {config.corpus_dir}[/red]")
        console.print("\nPlease create the directory and add some text files (.txt, .md, .pdf, .docx)")
        return
    
    files = list(corpus_path.rglob('*'))
    if not any(f.is_file() for f in files):
        console.print(f"[yellow]Warning: No files found in {config.corpus_dir}[/yellow]")
        console.print("\nAdd some text files to index.")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading embedding model...", total=None)
        pipeline = RAGPipeline(config)
        
        progress.update(task, description="Indexing documents...")
        num_chunks = pipeline.index_corpus()
    
    console.print(f"\n✅ [green]Successfully indexed {num_chunks} document chunks![/green]")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show system statistics."""
    config = ctx.obj['config']
    
    console.print("\n📊 [bold]System Statistics[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading pipeline...", total=None)
        pipeline = RAGPipeline(config)
    
    stats = pipeline.get_stats()
    
    
    table = Table(title="📚 Retriever", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats['retriever'].items():
        table.add_row(key, str(value))
    
    console.print(table)
    console.print()
    
    
    table = Table(title="🧮 Embedding Model", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats['embedding_model'].items():
        table.add_row(key, str(value))
    
    console.print(table)
    console.print()
    

    table = Table(title="🤖 Language Model", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats['llm'].items():
        table.add_row(key, str(value))
    
    console.print(table)


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='config.yaml', help='Output path')
@click.pass_context
def init_config(ctx, output):
    """Generate a default configuration file."""
    config = get_default_config()
    config.to_yaml(output)
    console.print(f"\n✅ Configuration file created: [cyan]{output}[/cyan]")
    console.print("\nEdit this file to customize your RAG system settings.")


def interactive_mode(config: RAGConfig):
    """Run the interactive chat mode."""
    print_banner()
    
    console.print("\n[dim]Loading RAG system... (this may take a moment)[/dim]\n")
    
    try:
        pipeline = RAGPipeline(config)
    except Exception as e:
        console.print(f"\n[red]Error initializing pipeline: {e}[/red]")
        console.print("\n[yellow]Tip: Make sure you have:[/yellow]")
        console.print("  1. Downloaded a GGUF model file")
        console.print("  2. Set the model path in config or use --model flag")
        return
    
 
    stats = pipeline.retriever.get_stats()
    if stats['num_documents'] == 0:
        console.print("\n[yellow]⚠️  No documents indexed yet![/yellow]")
        console.print("Run [cyan]rag-cli index[/cyan] to build the document index.")
        console.print("Or add files to the corpus directory and restart.\n")
    else:
        console.print(f"\n[green]✓[/green] {stats['num_documents']} documents ready for retrieval\n")
    
    console.print("[dim]Commands: 'quit' to exit, 'help' for help, 'stats' for statistics[/dim]\n")
    
    while True:
        try:
        
            query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if not query.strip():
                continue
            
           
            query_lower = query.lower().strip()
            
            if query_lower in ('quit', 'exit', 'q'):
                console.print("\n[dim]Goodbye! 👋[/dim]\n")
                break
            
            if query_lower == 'help':
                show_help()
                continue
            
            if query_lower == 'stats':
                show_stats(pipeline)
                continue
            
            if query_lower == 'clear':
                console.clear()
                print_banner()
                continue
            
           
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Thinking...", total=None)
                response = pipeline.query(query)
            
            print_response(response)
            
        except KeyboardInterrupt:
            console.print("\n\n[dim]Interrupted. Type 'quit' to exit.[/dim]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            if config.verbose:
                import traceback
                console.print(traceback.format_exc())


def show_help():
    """Show help information."""
    help_text = """
## Available Commands

| Command | Description |
|---------|-------------|
| `quit`, `exit`, `q` | Exit the program |
| `help` | Show this help message |
| `stats` | Show system statistics |
| `clear` | Clear the screen |

## Tips

- **Be specific**: More detailed questions get better answers
- **Check sources**: Review the source citations to verify information
- **Refine queries**: If results aren't relevant, try rephrasing

## Examples

- "What is machine learning?"
- "How does the retrieval system work?"
- "Explain the concept of embeddings"
    """
    console.print(Panel(Markdown(help_text), title="📖 Help", border_style="blue"))


def show_stats(pipeline: RAGPipeline):
    """Show current statistics."""
    stats = pipeline.get_stats()
    
    console.print(f"\n📊 **Statistics**")
    console.print(f"  Documents indexed: {stats['retriever']['num_documents']}")
    console.print(f"  Embedding dimension: {stats['embedding_model']['dimension']}")
    console.print(f"  Guardrails: {'enabled' if stats['guardrails_enabled'] else 'disabled'}")


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()

