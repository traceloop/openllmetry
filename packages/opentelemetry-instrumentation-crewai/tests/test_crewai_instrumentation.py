import pytest
from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.test_utils import InstrumentationTest
from crewai import Agent, Task, Crew
from unittest.mock import MagicMock

class TestCrewAIInstrumentation(InstrumentationTest):
    def setUp(self):
        super().setUp()
        self.instrumentation = CrewAIInstrumentor()
        self.instrumentation.instrument()

    def tearDown(self):
        super().tearDown()
        self.instrumentation.uninstrument()

    def test_crew_kickoff(self):
        # Create mock LLM
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "Mocked response"

        # Create test agents and tasks
        agent = Agent(
            role="Test Agent",
            goal="Test Goal",
            backstory="Test Backstory",
            llm=mock_llm
        )

        task = Task(
            description="Test Task",
            agent=agent
        )

        crew = Crew(
            agents=[agent],
            tasks=[task]
        )

        # Execute crew
        crew.kickoff()

        # Verify spans
        spans = self.memory_exporter.get_finished_spans()
        assert len(spans) >= 2  # At least one for workflow and one for task

        workflow_span = spans[0]
        assert workflow_span.name == "crewai.workflow"
        assert workflow_span.attributes["ai.workflow.name"] == "crewai.workflow"

        task_span = spans[1]
        assert task_span.name == "crewai.task"
        assert task_span.attributes["ai.agent.role"] == "Test Agent"
        assert task_span.attributes["ai.task.description"] == "Test Task"

    def test_error_handling(self):
        mock_llm = MagicMock()
        mock_llm.predict.side_effect = Exception("Test error")

        agent = Agent(
            role="Test Agent",
            goal="Test Goal",
            backstory="Test Backstory",
            llm=mock_llm
        )

        task = Task(
            description="Test Task",
            agent=agent
        )

        crew = Crew(
            agents=[agent],
            tasks=[task]
        )

        with pytest.raises(Exception):
            crew.kickoff()

        spans = self.memory_exporter.get_finished_spans()
        assert spans[-1].status.status_code == StatusCode.ERROR 