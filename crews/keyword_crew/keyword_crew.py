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
class KeywordCrew:
    """Crew for extract keyword from a text"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    mcp_server_params = [
        {
            "url": f"http://localhost:{settings.mcp_proxy_words_port}/mcp", #proxy
            "transport": "streamable-http"
        }
    ]

    @agent
    def keyword_extractor(self) -> Agent:
        crew_llm = LLM(
            model=f"ollama/{settings.ollama_model}",
            base_url=settings.ollama_base_url,
            temperature=0.5
        )

        return Agent(
            config=self.agents_config['keyword_extractor'], 
            llm=crew_llm,
            tools=self.get_mcp_tools()
        )

    @task
    def extract_keywords(self) -> Task:
        return Task(
            config=self.tasks_config['extract_keywords'],
            agent=self.keyword_extractor(),
        )


    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, 
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

