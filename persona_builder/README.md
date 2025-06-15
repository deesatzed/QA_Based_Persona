# AI Persona Builder

This application uses AI to build a rich, consistent personality profile by analyzing a user's existing chat history. This profile can then be used to answer a series of introspective questions (like the "Thousand Questions") as that persona.

**Current Phase: Persona Profiling**

This initial version focuses on Step 1: Generating a persona profile from a chat history.

## Prerequisites

- Python 3.9+
- An OpenAI API key

## Setup

1.  **Clone the repository (or ensure these files are in your project directory).**

2.  **Navigate to the `persona_builder` project directory:**
    ```bash
    cd persona_builder
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up your environment variables:**
    Create a `.env` file in the root of the `persona_builder` project directory (alongside this README) and add your OpenAI API key:
    ```env
    OPENAI_API_KEY="your-openai-api-key-here"
    ```

6.  **Prepare your data:**
    - Place your chat history text file (e.g., `my_chat_history.txt`) in the `chat_histories/` directory. You can create this directory if it's not present.
    - The `Thousand_Questions.txt` file should be placed in the `source_data/` directory for later phases. An empty placeholder is currently there.

## Usage

The application is run from the command line from within the `persona_builder` directory.

### Step 1: Generate the Persona Profile

Run the `profile` command, providing the path to your chat history file.

```bash
python -m persona_builder.main profile chat_histories/your_chat_history.txt
```

For example, if you have `chat_histories/sample_chat.txt`:
```bash
python -m persona_builder.main profile chat_histories/sample_chat.txt
```

This will analyze the chat history and create a `[your_chat_history_stem]_persona_profile.json` file in the `outputs/` directory.

---
*The ability to answer questions based on this profile (the `answer` command) will be implemented in Phase 2.*
