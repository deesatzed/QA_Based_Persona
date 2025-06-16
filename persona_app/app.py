import streamlit as st
from .persona_agent import create_persona_agent  # noqa: E402
import os
from dotenv import load_dotenv
import nest_asyncio
import datetime

# Apply nest_asyncio to allow asyncio event loops to be nested, common in Streamlit apps using async libraries
nest_asyncio.apply()

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

st.set_page_config(page_title="PersonaChat", layout="wide")

st.title(" PersonaChat: Create and Chat with AI Personas")

# --- Helper Functions ---
def export_chat_history(persona_name, messages):
    """Exports chat history to a markdown file."""
    if not messages:
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{persona_name.replace(' ', '_')}_chat_{timestamp}.md"

    exported_content = f"# Chat History with {persona_name}\n\n"
    for message in messages:
        role = "User" if message["role"] == "user" else persona_name
        exported_content += f"**{role}:** {message['content']}\n\n"

    return filename, exported_content

# --- Session State Initialization ---
if "current_persona_name" not in st.session_state:
    st.session_state.current_persona_name = "DefaultPersona"
if "current_persona_description" not in st.session_state:
    st.session_state.current_persona_description = "A helpful AI assistant."
if "current_persona_instructions" not in st.session_state:
    st.session_state.current_persona_instructions = "Respond helpfully and concisely."
if "current_seed_history" not in st.session_state:
    st.session_state.current_seed_history = ""
if "messages" not in st.session_state:
    st.session_state.messages = [] # To store chat messages for the current session
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None # Will be set when agent is created/loaded

# --- Sidebar for Persona Definition ---
with st.sidebar:
    st.header("Define Your Persona")

    persona_name_input = st.text_input("Persona Name", value=st.session_state.current_persona_name)
    persona_description_input = st.text_area("Persona Description", value=st.session_state.current_persona_description, height=100)
    persona_instructions_input = st.text_area("Persona Instructions", value=st.session_state.current_persona_instructions, height=150)

    st.markdown("---")
    st.subheader("Seed with Chat History (Optional)")
    uploaded_file = st.file_uploader("Upload a .txt chat history file", type="txt")

    seed_history_display = st.session_state.current_seed_history
    if uploaded_file is not None:
        try:
            seed_history_display = uploaded_file.read().decode()
            st.text_area("Seed History Preview", seed_history_display, height=100, disabled=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            seed_history_display = ""


    if st.button("🔄 Update Persona & Start New Chat", use_container_width=True):
        if not persona_name_input.strip():
            st.error("Persona Name cannot be empty.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        else:
            st.session_state.current_persona_name = persona_name_input
            st.session_state.current_persona_description = persona_description_input
            st.session_state.current_persona_instructions = persona_instructions_input
            st.session_state.current_seed_history = seed_history_display

            # Reset chat and agent for the new persona
            st.session_state.messages = []
            st.session_state.session_id = None # Ensure a new session for the new persona

            try:
                st.session_state.active_agent = create_persona_agent(
                    persona_name=st.session_state.current_persona_name,
                    persona_description=st.session_state.current_persona_description,
                    persona_instructions=st.session_state.current_persona_instructions,
                    seed_history=st.session_state.current_seed_history,
                    session_id=st.session_state.session_id # Should be None to start new
                )
                st.session_state.session_id = st.session_state.active_agent.session_id # Get the new session ID
                st.success(f"Persona '{st.session_state.current_persona_name}' activated. Start chatting!")
                # Add a system message or initial greeting from the persona if desired
                st.session_state.messages.append({"role": "assistant", "content": f"Hello! I am {st.session_state.current_persona_name}. How can I help you today?"})

            except Exception as e:
                st.error(f"Failed to create persona agent: {e}")
                st.session_state.active_agent = None

    st.markdown("---")
    st.subheader("Chat Controls")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        # Optionally, also reset the agent's internal history if the library supports it
        # without creating a new agent object. For now, this clears UI display.
        if st.session_state.active_agent:
             # Re-create agent to clear memory for this session_id, or start new if session_id is None
            try:
                st.session_state.active_agent = create_persona_agent(
                    persona_name=st.session_state.current_persona_name,
                    persona_description=st.session_state.current_persona_description,
                    persona_instructions=st.session_state.current_persona_instructions,
                    seed_history=st.session_state.current_seed_history, # re-apply seed if desired
                    session_id=st.session_state.session_id # Use existing session_id to clear its history
                )
                # Add a system message or initial greeting from the persona if desired
                st.session_state.messages.append({"role": "assistant", "content": f"Chat history cleared. I am {st.session_state.current_persona_name}. How can I assist?"})

            except Exception as e:
                st.error(f"Failed to reset agent: {e}")

        st.rerun()

    # Export chat button
    if st.session_state.messages:
        file_name, file_content = export_chat_history(st.session_state.current_persona_name, st.session_state.messages)
        if file_name:
            st.download_button(
                label="Export Chat to Markdown",
                data=file_content,
                file_name=file_name,
                mime="text/markdown",
                use_container_width=True
            )

# --- Main Chat Interface ---
st.header(f"Chat with: {st.session_state.current_persona_name}")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("OPENAI_API_KEY is not set. Please configure it in a .env file in the project root and restart the app.")
elif not st.session_state.active_agent:
    st.info("Define and update/start a persona using the sidebar to begin chatting.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What do you want to say?"):
    if not st.session_state.active_agent:
        st.error("No active persona. Please define and update/start a persona in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                # For agno, the interaction might be direct method calls.
                # Assuming a simple 'chat' or 'invoke' method on the agent.
                # Adjust based on 'agno' library's specific API.
                # The `persona_agent.py` currently returns an `Agent` object.
                # We need to know how to send a message and get a response.
                # Let's assume agent.chat(prompt) or agent.invoke(prompt)

                # Placeholder for how agno might stream or return response:
                # This is a guess. `agno` documentation would be needed.
                # For synchronous response:
                response_data = st.session_state.active_agent.chat(prompt) # Or .invoke, .ask, etc.

                # If response_data is a string:
                full_response = response_data

                # If response_data is an object with a 'content' attribute:
                # full_response = response_data.content

                # If agno supports streaming (conceptual):
                # for chunk in st.session_state.active_agent.stream(prompt):
                #    full_response += chunk.get("content", "")
                #    message_placeholder.markdown(full_response + "▌")
                # message_placeholder.markdown(full_response)

            except AttributeError as e:
                full_response = f"Error: The 'active_agent' does not have the expected chat method. ({e})"
            except Exception as e:
                full_response = f"Error interacting with agent: {e}"

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
