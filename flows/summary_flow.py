import logging
from typing import List

from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from crews.summary_crew.summary_crew import SummaryCrew

from utils.flow_helpers import handle_exceptions
from langchain.schema import Document

class SummaryState(BaseModel):
    documents: List[Document] = []
    full_text: str = ""
    instructions: str = ""
    summary_type: str = "comprehensive"
    summary_text: str = ""
    num_documents: int = 0
    sources: list = []
    word_count: int = 0

logger = logging.getLogger(__name__)

class SummaryFlow(Flow[SummaryState]):

    @start()
    @handle_exceptions
    def get_the_documents(self):
        logger.debug(f"Starting summary flow")

        if not self.state.documents:
            return {
                "success": False,
                "summary": "No documents available to summarize",
                "summary_type": self.state.summary_type
            }
            
        self.state.full_text = "\n\n".join([doc.page_content for doc in self.state.documents[:20]])
        
        return { 
            "documents": self.state.documents,
            "full_text": self.state.full_text
        }
    
    @listen(get_the_documents)
    @handle_exceptions
    def get_instructions(self):
        logger.debug("get_instructions")

        if self.state.summary_type == "brief":
            self.state.instructions = "Provide a brief 3-5 sentence summary of the key points."
        elif self.state.summary_type == "executive":
            self.state.instructions = """Provide an executive summary with:
            - Main objective/purpose
            - Key findings (3-5 points)
            - Critical numbers or metrics
            - Recommendations or implications"""
        else:  # comprehensive
            self.state.instructions = """Provide a comprehensive summary including:
            - Document overview and purpose
            - Main topics covered
            - Key findings and insights
            - Important financial metrics, numbers, or data points
            - Notable trends or patterns
            - Conclusions or implications"""
        return { "instructions": self.state.instructions}
        

    @listen(get_instructions)
    @handle_exceptions
    def generate_summary(self):
        logger.debug("generate_summary")

        result = SummaryCrew().crew().kickoff(inputs={
            "full_text": self.state.full_text,
            "instructions": self.state.instructions,
            "summary_type": self.state.summary_type
        })

        self.state.summary_text = str(result)
        return { "summary_text": self.state.summary_text }
        

    @listen(generate_summary)
    @handle_exceptions
    def get_sources(self):
        logger.debug("get_sources")

        self.state.sources = list(set([doc.metadata.get('source', 'Unknown') for doc in self.state.documents]))
        return { "summary_text": self.state.sources }
    
    @listen(generate_summary)
    @handle_exceptions
    def get_word_count(self):
        logger.debug("get_word_count")

        self.state.word_count = len(str(self.state.word_count).split())
    
        return {
                "success": True,
                "summary": str(self.state.summary_text),
                "summary_type": self.state.summary_type,
                "num_documents": len(self.state.documents),
                "sources": self.state.sources,
                "word_count": self.state.word_count
            }

