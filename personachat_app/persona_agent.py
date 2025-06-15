from pathlib import Path
from textwrap import dedent
from typing import Optional

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage

# Create a temporary directory for the database
tmp_dir = Path(__file__).parent.joinpath("tmp")
tmp_dir.mkdir(parents=True, exist_ok=True)

# Use SQLite for storing agent sessions
agent_storage = SqliteAgentStorage(
    table_name="persona_agent_sessions",
    db_file=str(tmp_dir.joinpath("personas.db")),
)


def create_persona_agent(
    persona_name: str,
    persona_description: str,
    persona_instructions: str,
    seed_history: Optional[str] = None,
    model_id: str = "openai:gpt-4o",
    session_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Agent:
    """
    Creates a new Agent instance or loads an existing one based on session_id.

    The agent is configured with a persona (name, description, instructions)
    and can be seeded with a chat history.
    """

    system_prompt = dedent(
        f"""
        You are {persona_name}.
        {persona_description}
        Instructions: {persona_instructions}
        """
    )

    if seed_history:
        system_prompt += f"\n\n--- Seed Chat History ---\n{seed_history}\n--- End of Seed Chat History ---"

    # TODO: The 'agno' library's Agent creation or model interaction might need specific handling
    # for system prompts and seed history. The following is a plausible interpretation based on typical agent frameworks.
    # Adjust if 'agno' handles this differently.

    agent = Agent(
        model=OpenAIChat(model_id=model_id, system_prompt=system_prompt),
        storage=agent_storage,
        session_id=session_id,  # If None, a new session will typically be created
        debug=debug_mode,
    )

    # If it's a new session and seed_history is provided, some libraries might require
    # adding the seed history as initial messages.
    # Example (conceptual, may need adjustment for agno):
    # if not agent.history() and seed_history: # Check if history is empty
    #     # This part is highly dependent on how 'agno' expects seed history to be injected.
    #     # It might be through the system prompt (as above), or by adding messages.
    #     # For now, assuming the system prompt handles it.
    #     pass

    return agent
