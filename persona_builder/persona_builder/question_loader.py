import logging
from typing import List, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThousandQuestionsLoader:
    '''Loads and categorizes the thousand questions from the source file.'''

    def __init__(self, questions_file_path: Path):
        self.questions_file = questions_file_path
        self.questions: List[Dict] = []
        self.categories: Dict[str, List[Dict]] = {}
        self._load_questions()

    def _load_questions(self):
        '''Loads and parses questions from the text file.'''
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            current_category = "Uncategorized"
            question_id = 1
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Category headers are not indented and don't end with a '?'
                if not line.startswith((' ', '\t')) and not line.endswith('?'):
                    current_category = line
                    if current_category not in self.categories:
                        self.categories[current_category] = []
                    logger.debug(f"Switched to category: {current_category}")
                    continue

                # Question lines are indented or end with a '?'
                if line.endswith('?'):
                    question_text = line.lstrip() # Remove leading whitespace
                    question = {
                        "id": question_id,
                        "text": question_text,
                        "category": current_category
                    }
                    self.questions.append(question)
                    if current_category in self.categories:
                        self.categories[current_category].append(question)
                    else:
                        logger.warning(f"Question '{question_text}' added to '{current_category}' but category was not pre-initialized.")
                        self.categories[current_category] = [question]
                    question_id += 1

            if not self.questions:
                logger.warning(f"No questions loaded from {self.questions_file}. The file might be empty or formatted incorrectly.")
            else:
                logger.info(f"Loaded {len(self.questions)} questions across {len(self.categories)} categories from {self.questions_file}.")

        except FileNotFoundError:
            logger.error(f"FATAL: Questions file not found at: {self.questions_file}")
        except Exception as e:
            logger.error(f"Error loading questions from {self.questions_file}: {e}")
