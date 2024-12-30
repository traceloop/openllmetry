import os
from openai import OpenAI

from traceloop.sdk import Traceloop

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Traceloop.init()


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What's the summary of PR #234",
        },
    ],
    temperature=0,
    user="user-123456",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "summarize_github_pr_content",
                "description": "summarize a github pull request by url",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "github_pull_request_url": {
                            "type": "string",
                            "description": "The GitHub pull request url",
                        }
                    },
                    "required": ["github_pull_request_url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "github_fetcher",
                "description": "Gets the code commits for a github repository and a specific owner. if you don't"
                + " have the required parameters in the specification, you need to ask the user to provide them",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "owner": {
                            "type": "string",
                            "description": "The owner of the github repository",
                        },
                        "repo": {
                            "type": "string",
                            "description": "The github repository name",
                        },
                        "since": {
                            "type": "string",
                            "description": "Only show notifications updated after the given time. This is a timestamp "
                            + "in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ.",
                        },
                    },
                    "required": ["owner", "repo"],
                },
            },
        },
    ],
)

print(response)
