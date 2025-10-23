"""
Console UI for model selection and training using rich formatting.
Uses the 'rich' library for attractive console output.
"""

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

console = Console()

def show_menu():
    table = Table(title="Neural Network Model Selection", show_header=True, header_style="bold magenta")
    table.add_column("Option", style="cyan", width=8)
    table.add_column("Model", style="green")
    table.add_row("1", "MLP (Multilayer Perceptron, numpy)")
    table.add_row("2", "RNN (Keras LSTM)")
    table.add_row("3", "CNN (Keras Conv2D)")
    console.print(table)
    choice = Prompt.ask("[bold yellow]Enter option (1, 2, or 3)[/bold yellow]", choices=["1", "2", "3"])
    return choice

def show_welcome():
    console.print(Panel(Text("Welcome to the Neural Network Playground!", justify="center", style="bold white on blue")))

def show_result(message):
    console.print(Panel(Text(message, justify="center", style="bold green")))
