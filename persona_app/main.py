import typer
import os
import json
import logging
from pathlib import Path

from .persona_profiler import PersonaProfiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer(help='AI Persona Builder CLI.', add_completion=False)

BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "outputs"
CHAT_HISTORIES_DIR = BASE_DIR / "chat_histories"
SOURCE_DATA_DIR = BASE_DIR / "source_data"

@app.callback()
def main_callback(ctx: typer.Context):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ensured: {OUTPUT_DIR.resolve()}")

@app.command()
def profile(
    chat_history_file: Path = typer.Argument(
        ...,
        help='Path to the chat history file (e.g., chat_histories/sample_chat.txt).',
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    )
):
    logger.info(f"Starting Persona Profiling for: '{chat_history_file}'")
    try:
        with open(chat_history_file, 'r', encoding='utf-8') as f:
            chat_history = f.read()
        if not chat_history.strip():
            logger.error(f"Chat history file '{chat_history_file}' is empty.")
            typer.echo(f"Error: Chat history file '{chat_history_file}' is empty.")
            raise typer.Exit(code=1)

        profiler = PersonaProfiler()
        persona_profile_data = profiler.create_persona_from_chat_history(chat_history)

        if persona_profile_data is None:
            logger.error("Profiler returned None. Persona profile generation failed.")
            typer.echo("Error: Failed to generate persona profile. Check logs.")
            raise typer.Exit(code=1)

        profile_filename = f"{chat_history_file.stem}_persona_profile.json"
        output_path = OUTPUT_DIR / profile_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(persona_profile_data, f, indent=2)

        success_message = f"Persona profile saved to: '{output_path.resolve()}'"
        logger.info(success_message)
        typer.echo(f"✅ {success_message}")

    except FileNotFoundError:
        logger.error(f"File not found: '{chat_history_file}'.")
        typer.echo(f"Error: Chat history file not found at '{chat_history_file}'.")
        raise typer.Exit(code=1)
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        typer.echo(f"Error: {ve}. Is OPENAI_API_KEY set in .env?")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Unexpected error during profiling: {e}", exc_info=True)
        typer.echo(f"An unexpected error occurred: {e}. Check logs.")
        raise typer.Exit(code=1)

@app.command()
def answer(
    persona_profile_path: Path = typer.Argument(
        ...,
        help='Path to the generated persona_profile.json.',
        exists=True,
        readable=True,
        resolve_path=True
    ),
    questions_file: Path = typer.Option(
        lambda: SOURCE_DATA_DIR / "Thousand_Questions.txt",
        help='Path to the Thousand_Questions.txt file.',
        exists=True,
        readable=True,
        resolve_path=True
    )
):
    logger.info(f"Placeholder 'answer' command. Persona: {persona_profile_path}, Questions: {questions_file}")
    if not questions_file.exists():
        typer.echo(f"Warning: Default questions file {questions_file} not found.")
    typer.echo("The 'answer' command is not yet fully implemented (Phase 2).")

if __name__ == "__main__":
    app()
