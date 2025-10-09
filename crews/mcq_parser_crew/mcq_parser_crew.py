import logging
from crewai import Agent, Crew, Task, Process, LLM
from crewai.project import CrewBase, agent, task, crew, tool
from crewai.agents.agent_builder.base_agent import BaseAgent

from tools.mcq_parser_tool import mcqs_parser_tool
from typing import List

from config.settings import settings
from utils.custom_listener import MyCustomListener
from pprint import pprint

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

my_listener = MyCustomListener()

@CrewBase
class MCQParserCrew:
    """Crew to structure raw MCQs into clean, parseable format"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @tool
    def mcqs_parser_tool(self):
        return mcqs_parser_tool
   
    @agent
    def mcq_parser_agent(self) -> Agent:
        crew_llm = LLM(
            model=f"ollama/{settings.ollama_model}",
            base_url=settings.ollama_base_url,
            temperature=0.0
        )

        return Agent(
            config=self.agents_config['mcq_parser_agent'], 
            llm=crew_llm
        )

    @task
    def parse_mcqs_task(self) -> Task:
        return Task(
            config=self.tasks_config['parse_mcqs_task'],
            agent=self.mcq_parser_agent()
        )


    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, 
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            prompt_file="crews/mcq_parser_crew/custom_prompts.json"
        )

