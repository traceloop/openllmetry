import os
import openai

from traceloop.sdk import Traceloop

openai.api_key = os.getenv("OPENAI_API_KEY")
Traceloop.init()


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What's the summary of PR #234",
        },
    ],
    temperature=0,
    request_timeout=5,
    user="user-123456",
    headers={"x-session-id": "abcd-1234-cdef"},
    functions=[
        {
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
        {
            "name": "github_fetcher",
            "description": "Gets the code commits for a github repository and a specific owner. if you don't have the required parameters in the specification, you need to ask the user to provide them",
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
                        "description": "Only show notifications updated after the given time. This is a timestamp in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ.",
                    },
                },
                "required": ["owner", "repo"],
            },
        },
    ],
)

print(response)
