from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from traceloop.sdk import Traceloop

Traceloop.init(app_name="crewai-example")

llm = ChatOpenAI(model="gpt-4o-mini")

researcher = Agent(
    role='Research Analyst',
    goal='Conduct detailed market research',
    backstory='Expert in market analysis with keen attention to detail',
    llm=llm,
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Create engaging content based on research',
    backstory='Experienced writer with expertise in creating compelling narratives',
    llm=llm,
    verbose=True
)

research_task = Task(
    description='Research the latest trends in AI and machine learning',
    agent=researcher,
    expected_output='A comprehensive report on current AI and machine learning trends',
    expected_output_type=str
)

writing_task = Task(
    description='Write a blog post about AI trends based on the research',
    agent=writer,
    expected_output='An engaging blog post covering the latest AI trends',
    expected_output_type=str
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

result = crew.kickoff()
print("Final Result:", result)
