# Persona App

This repository combines the previous `persona_builder` and `personachat_app` projects.
It provides a command line tool for generating persona profiles from chat histories
and a Streamlit interface for chatting with those personas.

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your API key in a `.env` file in the project root:
   ```env
   OPENAI_API_KEY="your-openai-api-key"
   ```

## Generate a Persona Profile

Place a chat history text file somewhere on disk and run:
```bash
python -m persona_app.main profile path/to/chat_history.txt
```
The generated profile JSON will be written to the `outputs/` directory.

## Launch the Chat Interface

Start the Streamlit application with:
```bash
streamlit run persona_app/app.py
```
Then open `http://localhost:8501` in your browser to chat with your personas.
