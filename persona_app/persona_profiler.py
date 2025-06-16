import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict
from pathlib import Path # Added for the example

# Configure logging and environment
load_dotenv() # Ensure .env is loaded for OPENAI_API_KEY
logger = logging.getLogger(__name__)

class PersonaProfiler:
    '''
    Uses an AI model to analyze a chat history and create a persona profile.
    Initial version based on the "AI Persona Builder" specification.
    '''

    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided or found in .env file.")
        self.client = OpenAI(api_key=api_key)
        self.model_id = "gpt-4o" # As specified in the document for complex analysis

    def _get_profiling_prompt_template(self) -> str:
        '''
        Returns the system prompt template for the persona profiling task.
        This is the structured JSON format expected from the LLM.
        '''
        return '''
You are a Persona Architect AI. Your task is to analyze a provided chat history and create a rich, nuanced, and structured personality profile in JSON format. This profile will serve as the foundation for a simulated AI agent.

Based on the chat history, infer the user's personality. Analyze their language, topics of discussion, expressed opinions, values, emotions, and communication style.

Your output MUST be a JSON object with the following structure:

{
  "name": "A plausible name for this persona (e.g., 'The Pragmatic Tinkerer', 'The Compassionate Explorer')",
  "role": "A professional or archetypal role (e.g., 'Data-Driven Experimentalist', 'Multidisciplinary Ethicist')",
  "specialty": "A specific area of expertise or interest evident from the chats (e.g., 'AI safety, normative philosophy', 'Distributed systems')",
  "personality_summary": "A 2-3 paragraph narrative summary of the persona. Describe their core motivations, how they see the world, and their general demeanor. This should be a holistic synthesis.",
  "core_traits": {
    "analytical_thinking": "A float from 0.0 to 1.0 representing their tendency for logical, analytical thought.",
    "intellectual_curiosity": "A float from 0.0 to 1.0 for their desire to learn and explore new ideas.",
    "practical_focus": "A float from 0.0 to 1.0 indicating a focus on practical application vs. theory.",
    "empathetic_understanding": "A float from 0.0 to 1.0 for their ability to understand and share the feelings of others.",
    "risk_awareness": "A float from 0.0 to 1.0 for their tendency to consider risks and act cautiously."
  },
  "knowledge_domains": [
    "A list of 5-10 specific topics or fields the user seems knowledgeable about (e.g., 'machine_learning', 'philosophy_of_mind', 'gardening')."
  ],
  "communication_style": {
    "formality": "A float from 0.0 (very informal) to 1.0 (very formal).",
    "technical_depth": "A float from 0.0 (explains simply) to 1.0 (uses deep technical jargon).",
    "precision": "A float from 0.0 (vague, general) to 1.0 (highly precise and specific).",
    "collaborative": "A float from 0.0 (independent, declarative) to 1.0 (highly collaborative, asks questions)."
  }
}

Analyze the provided chat history carefully and populate every field in the JSON structure. Be nuanced and base your analysis strictly on the evidence in the text.
'''

    def create_persona_from_chat_history(self, chat_history: str) -> Dict:
        '''
        Analyzes a chat history to generate a structured persona profile.

        Args:
            chat_history (str): The full text content of the user's chat history.

        Returns:
            dict: A dictionary representing the structured persona profile.
                  Returns None if an error occurs during API call or JSON parsing.
        '''
        system_prompt = self._get_profiling_prompt_template()
        logger.info(f"Sending chat history to {self.model_id} for persona profiling...")

        response_content = None # Initialize for robust error logging
        try:
            # Constructing the user message carefully
            user_message_content = f"Here is the user's chat history:\n\n---\n\n{chat_history}"

            response = self.client.chat.completions.create(
                model=self.model_id,
                response_format={"type": "json_object"}, # Ensure JSON output
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message_content}
                ]
            )

            response_content = response.choices[0].message.content
            persona_profile = json.loads(response_content)
            logger.info("Successfully generated persona profile.")
            return persona_profile

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response into a valid persona profile: {e}")
            if response_content:
                logger.error(f"Raw AI Response:\n{response_content}")
            else:
                logger.error("No response content received from AI.")
            return None
        except Exception as e:
            logger.error(f"An error occurred during OpenAI API call: {e}")
            if response_content: # Log content if available even on other API errors
                 logger.error(f"Raw AI Response (at time of different error):\n{response_content}")
            return None

if __name__ == '__main__':
    # Example Usage (requires OPENAI_API_KEY in .env or passed to constructor)
    # This part is for basic testing and will not be part of the final package execution.
    logger.info("Persona Profiler Example Usage (requires .env file with OPENAI_API_KEY)")

    # Create a dummy .env if it doesn't exist for this example to run without error,
    # though it won't work without a real key.
    # Note: In a real subtask environment, file system operations like this might be restricted
    # or behave differently. This example is primarily for local developer testing.
    env_file = Path(".env")
    if not env_file.exists() and not os.getenv("OPENAI_API_KEY"):
        try:
            with open(env_file, "w") as f:
                f.write("OPENAI_API_KEY=\"your_key_here_if_not_set_externally\"\n")
            logger.info(f"Created a dummy {env_file}. Please ensure OPENAI_API_KEY is set.")
        except Exception as e:
            logger.warning(f"Could not create dummy .env file: {e}")


    load_dotenv(override=True) # Load the dummy or real .env, override ensures it's re-read

    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_key_here_if_not_set_externally":
        logger.warning("OPENAI_API_KEY not found or is a dummy value. API call in example will likely fail or be skipped.")

    # Proceed with example only if a potentially valid key is found
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_key_here_if_not_set_externally":
        try:
            profiler = PersonaProfiler()
            sample_chat_history = (
                "User: Hi there, I'm looking into learning about sustainable gardening. Any tips?\n"
                "AI: Hello! Sustainable gardening is a great topic. You could start with companion planting and composting.\n"
                "User: Tell me more about composting. Is it hard?\n"
                "AI: Not at all! You can start a simple compost pile with kitchen scraps and yard waste. It's great for your soil.\n"
                "User: I'm also very interested in philosophy, especially ethics in AI. It's a complex field.\n"
                "AI: Indeed, AI ethics is a rapidly evolving and critical area of discussion."
            )

            logger.info("Processing sample chat history...")
            profile = profiler.create_persona_from_chat_history(sample_chat_history)

            if profile:
                logger.info("Generated Profile:")
                # Using logger.info for each line of JSON to ensure it's captured if stdout is limited
                for line in json.dumps(profile, indent=2).splitlines():
                    logger.info(line)
            else:
                logger.error("Failed to generate profile from sample history.")

        except ValueError as ve:
            logger.error(f"Setup Error in example: {ve}")
        except Exception as ex:
            logger.error(f"An unexpected error occurred in example: {ex}")
    else:
        logger.info("Skipping profiler example API call due to missing or dummy API key.")
