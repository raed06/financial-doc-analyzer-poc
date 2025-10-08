import logging
from crewai import Agent, Crew, Task, LLM
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from config.settings import settings
from utils.llm_manager import LLMManager
from utils.vector_store_manager import VectorStoreManager

# Initialize logger
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

@CrewBase
class QACrew:
    """Crew for answering questions based on financial documents using RAG"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, vector_manager: VectorStoreManager):
        self.vector_manager = vector_manager
        self.last_inputs = {}

    @before_kickoff
    def prepare_inputs(self, inputs):
        logger.info("Preparing inputs for crew execution.")
        question = inputs.get("question", "")
        
        try:
            relevant_docs = self.vector_manager.similarity_search(question, k=4)
            self.last_inputs["relevant_docs"] = relevant_docs;

            if not relevant_docs:
                logger.warning("No relevant documents found for question: %s", question)
                inputs["context"] = "No relevant documents found."
            else:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                logger.info("Context prepared with %d documents.", len(relevant_docs))
                inputs["context"] = context

                sources = [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "type": doc.metadata.get("type", "unknown"),
                        "page": doc.metadata.get("page", None)
                    }
                    for doc in relevant_docs
                ]
                self.last_inputs["sources"] = sources
        except Exception as e:
            logger.error("Error during vector search: %s", str(e))
            inputs["context"] = "Error retrieving context."
        
        return inputs

    @after_kickoff
    def process_output(self, output):
        logger.info(f"Processing output from agent")

        try: 
            relevant_docs = self.last_inputs.get("relevant_docs", [])
            tasks_output = output.dict().get("tasks_output", [])

            answer = ""
            keywords = None

            for task in tasks_output:
                task_name = task.get("name")
                task_raw = task.get("raw", "")

                if task_name == "answer_financial_question":
                    answer = task_raw

                elif task_name == "extract_keywords":
                    keywords = [kw.strip() for kw in task_raw.split(",") if kw.strip()]

            return {
                "success": True,
                "answer": answer,
                "keywords": keywords,
                "sources": self.last_inputs.get("sources", []),
                "confidence": "high" if len(relevant_docs) >= 3 else "medium",
                "num_sources": len(relevant_docs)
            }
        except Exception as e:
            logger.error("Error in process_output: %s", str(e))
            return {
                "success": False,
                "answer": "An error occurred during post-processing.",
                "sources": [],
                "confidence": "low"
            }

    @agent
    def financial_qa_specialist(self) -> Agent:
        crew_llm = LLM(
            model=f"ollama/{settings.ollama_model}",
            base_url=settings.ollama_base_url,
            temperature=0.7
        )

        return Agent(
            config=self.agents_config['financial_qa_specialist'], 
            verbose=True,
            allow_delegation=False,
            llm=crew_llm
        )
    
    @task
    def answer_financial_question(self) -> Task:
        return Task(
            config=self.tasks_config['answer_financial_question'] 
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=True
        )
    