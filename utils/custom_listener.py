from crewai.events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    AgentExecutionCompletedEvent,
    TaskCompletedEvent,
    ToolUsageFinishedEvent
)
from crewai.events import BaseEventListener

class MyCustomListener(BaseEventListener):
    def __init__(self):
        super().__init__()

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source, event):
            print(f"Crew '{event.crew_name}' has started execution!")

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_completed(source, event):
            print(f"Crew '{event.crew_name}' has completed execution!")
            print(f"Output: {event.output}")

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def on_agent_execution_completed(source, event):
            print(f"Agent '{event.agent.role}' completed task")
            print(f"Output: {event.output}")

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completion(source, event):
            print(f"Task '{event.task.name}' completed task")
            print(f"Output: {event.output}")

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_completion(source, event):
            print(f"Tool '{event.tool.name}' completed task")
            print(f"Output: {event.output}")