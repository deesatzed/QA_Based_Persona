# PersonaChat: Create and Chat with AI Personas

PersonaChat is a Streamlit application that allows you to create, customize, and interact with AI-powered personas. You can define an agent's personality through a name, a detailed description, and specific instructions.

A key feature of PersonaChat is the ability to "seed" a persona with an example chat history. By providing a sample conversation, you can guide the agent's tone, style, and voice, making the interaction feel more natural and aligned with the persona you envision.

## 🌟 Features

-   **Customizable Personas**: Define the name, description, and instructions for your AI agent.
-   **Seed with Chat History**: Upload a text file (.txt) with a sample conversation to give your persona a unique voice.
-   **Multiple LLM Providers**: Primarily designed for OpenAI models through the `agno` library, but `agno` may support others.
-   **Session Management**: Automatically saves your conversations using SQLite.
-   **Chat History**: Maintains the context of your conversation with each persona within a session.
-   **Export Conversations**: You can export your chat history as a markdown file.

## 🚀 Quick Start

### 1. Environment Setup

It's recommended to use a virtual environment.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### 2. Install Dependencies

Install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

You'll need an API key from an LLM provider. For OpenAI, create a `.env` file in the `personachat_app` project root directory (i.e., alongside `app.py`) and add your key:

```env
OPENAI_API_KEY="your_openai_api_key_here"
```

### 4. Launch the Application

Run the Streamlit application from your terminal, ensuring you are in the `personachat_app` directory:

```bash
streamlit run app.py
```

Now you can open your browser and navigate to `http://localhost:8501` to start creating and chatting with your AI personas!

## ⚙️ Project Structure

```
personachat_app/
├── .env             # API keys (you need to create this)
├── app.py           # Main Streamlit application
├── persona_agent.py # Logic for creating and managing AI persona agents
├── requirements.txt # Python dependencies
├── tmp/             # Directory for SQLite database and other temporary files
│   └── personas.db  # SQLite database for session storage (auto-created)
└── README.md        # This file
```

## 📝 Notes

- The `agno` library is used for agent creation and interaction. Ensure it's compatible with your chosen LLM and that its specific API for sending/receiving messages is correctly implemented in `app.py` if you modify the agent interaction logic.
- Seed history is currently implemented by appending it to the system prompt. The effectiveness might vary based on the LLM and the length of the seed history.
- Sessions are stored in an SQLite database (`tmp/personas.db`). Each time you "Update Persona & Start New Chat" with the same persona details, you can either continue a session if `session_id` management in `persona_agent.py` is adapted for loading, or it will start a new one. The current `app.py` starts a new session upon clicking "Update Persona & Start New Chat".
