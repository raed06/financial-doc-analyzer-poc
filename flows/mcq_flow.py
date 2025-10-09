import logging
import json
import re
from typing import Dict, Any, List

from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from crews.mcq_crew.mcq_crew import MCQCrew
from crews.mcq_parser_crew.mcq_parser_crew import MCQParserCrew

from utils.flow_helpers import handle_exceptions
from langchain.schema import Document

class MCQState(BaseModel):
    documents: List[Document] = []
    full_text: str = ""
    difficulty_instructions: str = ""
    questions_text: str = ""
    questions: List[Dict[str, Any]] = []
    num_questions: int = 5
    difficulty: str = "medium"

logger = logging.getLogger(__name__)

class MCQFlow(Flow[MCQState]):

    @start()
    @handle_exceptions
    def get_the_documents(self):
        logger.debug(f"Starting mcq flow")

        if not self.state.documents:
            return {
                "success": False,
                "questions": [],
                "message": "No documents available"
            }
        
        self.state.full_text = "\n\n".join([doc.page_content for doc in self.state.documents[:15]])
        
        return { 
            "documents": self.state.documents,
            "full_text": self.state.full_text
        }
    
    @listen(get_the_documents)
    @handle_exceptions
    def get_instructions(self):
        logger.debug("get_instructions")

        difficulty_map = {
            "easy": "Focus on basic facts and direct information from the text",
            "medium": "Test understanding and ability to apply concepts",
            "hard": "Require analysis, synthesis, and deep understanding"
        }
        self.state.difficulty_instructions = difficulty_map.get(self.state.difficulty, '')
        return { "instructions": self.state.difficulty_instructions }
        

    @listen(get_instructions)
    @handle_exceptions
    def generate_mcqs(self):
        logger.debug("generate_mcqs")
        
        result = MCQCrew().crew().kickoff(inputs={
            "full_text": self.state.full_text,
            "num_questions": self.state.num_questions,
            "difficulty": self.state.difficulty,
            "difficulty_instructions": self.state.difficulty_instructions,
        })

        self.state.questions_text = str(result)
        return { "questions_text": self.state.questions_text }
        

    @listen(generate_mcqs)
    @handle_exceptions
    def parse_mcqs(self):
        logger.debug("parse_mcqs")
        
        if not self.state.questions_text:
            logger.warning("No questions_text found in state")
            return {
                "success": False,
                "questions": [],
                "message": "No generated questions to parse."
            }

        result = MCQParserCrew().crew().kickoff(inputs={
            "raw_text": self.state.questions_text
        })

        raw = result.raw
        raw = raw.strip()
        match = re.search(r"(\[.*\])", raw, flags=re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            json_text = raw

        all_questions = json.loads(json_text)
        self.state.questions = all_questions[:self.state.num_questions]

        return {
            "success": True,
            "questions": self.state.questions,
            "num_questions": len(self.state.questions),
            "difficulty": self.state.difficulty
        }

    
    @listen(parse_mcqs)
    def get_sources(self):
        logger.debug("get_sources")

        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in self.state.documents]))

        return {
            "success": True,
            "questions": self.state.questions,
            "num_questions": len(self.state.questions),
            "difficulty": self.state.difficulty,
            "sources": sources
        }
    
