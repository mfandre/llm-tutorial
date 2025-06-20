from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.rule import Rule

from models import DeepResearchWithReasoning

def pretty_print_report(report: DeepResearchWithReasoning):
    console = Console()

    console.print(Rule(title="📋 Deep Research Report", style="bold blue"))

    console.print(Panel.fit(
        Text(report.question_summary, style="bold yellow"),
        title="❓ Question Summary",
        border_style="yellow"
    ))

    console.print(Panel.fit(
        Markdown("\n".join(f"1. {step}" if not step.strip().startswith("1.") else step for step in report.research_steps)),
        title="🔍 Research Steps",
        border_style="cyan"
    ))

    console.print(Panel.fit(
        Markdown("\n".join(f"- {finding}" for finding in report.key_findings)),
        title="📌 Key Findings",
        border_style="magenta"
    ))

    console.print(Panel.fit(
        Text(report.synthesis, style="bold white"),
        title="🧠 Synthesis",
        border_style="green"
    ))

    console.print(Panel.fit(
        Text(report.reasoning, style="italic white"),
        title="🧩 Reasoning",
        border_style="blue"
    ))

    console.print(Panel.fit(
        Text(report.final_answer, style="bold bright_green"),
        title="✅ Final Answer",
        border_style="bright_green"
    ))

    console.print(Rule(style="blue"))
