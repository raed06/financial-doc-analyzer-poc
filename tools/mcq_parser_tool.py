import re
import logging
from crewai.tools import tool

logger = logging.getLogger(__name__)

@tool("MCQ Parser Tool")
def parse_mcqs_tool(raw_text: str) -> list:
    """
        Parses multiple-choice question (MCQ) text into a JSON structure.
        Parameters:
        - raw_text (str): The raw text containing the questions to be processed.
    """    
    questions = []

    question_blocks = re.split(r'\*{0,2}Question \d+:\*{0,2}', raw_text, flags=re.IGNORECASE)

    for idx, block in enumerate(question_blocks[1:], start=1):  
        try:
            question_match = re.search(r'^(.*?)(?=\nA\))', block, re.DOTALL)
            question_text = question_match.group(1).strip() if question_match else ""

            options = {}
            for letter in ['A', 'B', 'C', 'D']:
                pattern = rf'{letter}\)\s*(.*?)(?=\n[A-D]\)|\n\*\*Correct Answer|\n\*\*Explanation|\Z)'
                match = re.search(pattern, block, re.DOTALL)
                if match:
                    options[letter] = match.group(1).strip()

            answer_match = re.search(r'\*\*Correct Answer:\*\*\s*([A-D])', block, re.IGNORECASE)
            correct_answer = answer_match.group(1).upper() if answer_match else "A"

            explanation_match = re.search(r'\*\*Explanation:\*\*\s*(.*?)(?=\n\*\*Question|\Z)', block, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"

            if question_text and len(options) == 4:
                questions.append({
                    "question": question_text,
                    "options": options,
                    "correct_answer": correct_answer,
                    "explanation": explanation
                })
            else:
                logger.warning(f"Invalid question format in block {idx}")

        except Exception as e:
            logger.warning(f"Error parsing question block {idx}: {e}")
            continue

    if not questions:
        logger.warning("No questions parsed successfully. Adding fallback.")
        questions.append({
            "question": "Based on the document, what is the main topic discussed?",
            "options": {
                "A": "Financial performance",
                "B": "Market analysis",
                "C": "Risk assessment",
                "D": "Strategic planning"
            },
            "correct_answer": "A",
            "explanation": "This is a fallback question. Please check document content or formatting."
        })

    logger.info(f"Parsed {len(questions)} questions successfully")

    return questions
