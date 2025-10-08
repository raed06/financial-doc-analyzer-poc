import logging
from typing import Any

from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from crews.qa_crew.qa_crew import QACrew
from crews.keyword_crew.keyword_crew import KeywordCrew

from utils.vector_store_manager import VectorStoreManager

from utils.flow_helpers import handle_exceptions

logger = logging.getLogger(__name__)

class QAState(BaseModel):
    question: str = ""
    answer: str = ""
    sources: list[dict[str, Any]] = []
    confidence: str = "low"
    keywords: str = ""

class QAFlow(Flow[QAState]):

    def __init__(self, vector_manager: VectorStoreManager):
        super().__init__(QAState())  
        self.vector_manager = vector_manager

    @start()
    @handle_exceptions
    def get_the_question(self):
        logger.debug(f"Starting qa flow for {self.state.question}")
        
        return {"question": self.state.question}

    @listen(get_the_question)
    @handle_exceptions
    def answer_for_question(self):
        logger.debug("answer_for_question")

        result = QACrew(self.vector_manager).crew().kickoff(inputs={"question": self.state.question})

        if result['success']:
            self.state.answer = result['answer']
            self.state.sources = result['sources']
            self.state.confidence = result['confidence']
            return {
                "success": True,
                "answer": self.state.answer,
                "sources": self.state.sources,
                "confidence": self.state.confidence
            }
        else:
            return {
                "success": False,
                "answer": "An error occurred during post-processing.",
                "sources": [],
                "confidence": "low"
            }

    @listen(answer_for_question)
    @handle_exceptions
    def generate_keywords(self):
        logger.debug("generate_keywords")

        result = KeywordCrew().crew().kickoff(inputs={"answer": self.state.answer})
        self.state.keywords = result.raw

        return {
                "success": True,
                "answer": self.state.answer,
                "sources": self.state.sources,
                "confidence": self.state.confidence,
                "keywords": self.state.keywords
            }
