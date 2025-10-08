import logging
from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from config.settings import settings

# Initialize logger
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

@CrewBase
class MCQCrew:
    """Crew Multiple Choice Question Generation"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def mcq_specialist_finance(self) -> Agent:
        crew_llm = LLM(
            model=f"ollama/{settings.ollama_model}",
            base_url=settings.ollama_base_url,
            temperature=0.5
        )

        return Agent(
            config=self.agents_config['mcq_specialist_finance'], 
            llm=crew_llm
        )

    @task
    def generate_mcqs(self) -> Task:
        return Task(
            config=self.tasks_config['generate_mcqs'],
            agent=self.mcq_specialist_finance(),
        )


    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, 
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

