import time
from openai import OpenAI
from traceloop.sdk import Traceloop

Traceloop.init()

client = OpenAI()

assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-turbo-preview",
)

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
)

run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Jane Doe. The user has a premium account.",
)

while run.status in ["queued", "in_progress", "cancelling"]:
    time.sleep(1)  # Wait for 1 second
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

if run.status == "completed":
    messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    for data in messages.data:
        print(f"{data.role}: {data.content[0].text.value}")
else:
    print(run.status)
