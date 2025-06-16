import sys
from pathlib import Path

# Allow importing the package from the project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from persona_builder.question_loader import ThousandQuestionsLoader


def test_loads_questions_and_categories():
    questions_file = Path(__file__).resolve().parents[1] / "source_data" / "Thousand_Questions.txt"
    loader = ThousandQuestionsLoader(questions_file)

    # We expect five categories with three questions each
    assert len(loader.categories) == 5
    assert len(loader.questions) == 15

    for category, questions in loader.categories.items():
        # each question dict should reference the same category
        for q in questions:
            assert q["category"] == category


def test_question_ids_increment():
    questions_file = Path(__file__).resolve().parents[1] / "source_data" / "Thousand_Questions.txt"
    loader = ThousandQuestionsLoader(questions_file)

    ids = [q["id"] for q in loader.questions]
    assert ids == list(range(1, len(loader.questions) + 1))
