import json
from typing import Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from evaluator.evaluators.base import Evaluator
from evaluator.lib.llm_proxy import get_llm_client
from evaluator.models.evaluator_result import EvaluatorResult
from evaluator.models.evaluator_input import EvaluatorInput
from traceloop.sdk import Traceloop

Traceloop.init()

class LLMAsAJudgeInput(EvaluatorInput):
    messages: list[dict[str, str]]
    llm_config: dict
    provider: str
    variables: dict
    evaluation_id: str | None = None
    generation: str = ""

class LLMAsAJudge(Evaluator[LLMAsAJudgeInput]):
    def __init__(self) -> None:
        self.llm_client: OpenAI = get_llm_client()

    def evaluate(self, input: LLMAsAJudgeInput) -> EvaluatorResult:
        serialized_messages = self.serialize_messages(input.messages, input.variables)

        print("NOMI - serialized_messages", serialized_messages)

        try:
            resp = self.llm_client.chat.completions.create(
                model=input.llm_config.get("model", "gpt-4o"),
                messages=serialized_messages,
                stop=input.llm_config.get("stop", None),
                response_format=input.llm_config["response_format"],
                temperature=input.llm_config.get("temperature", 0),
                max_tokens=input.llm_config.get("max_tokens", 100),
                top_p=input.llm_config.get("top_p", 1),
                presence_penalty=input.llm_config.get("presence_penalty", 0),
                frequency_penalty=input.llm_config.get("frequency_penalty", 0),
            )

            print("NOMI - resp", resp)

            result_dict = resp.choices[0].message.model_dump()
            result_content = result_dict["content"]

            print("NOMI - result_content", result_content)

            return EvaluatorResult(result=json.loads(result_content))
        except Exception as e:
            return EvaluatorResult(result={"error": str(e)})

    def serialize_messages(self, messages: list[dict[str, str]], variables: dict[str, Any]) -> list[ChatCompletionMessageParam]:
        serialized_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            content = msg["content"]
            for var_name, var_value in variables.items():
                content = content.replace(f"{{{{{var_name}}}}}", str(var_value))

            role = msg["role"]
            if role == "system":
                serialized_messages.append({"role": "system", "content": content})
            elif role == "user":
                serialized_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                serialized_messages.append({"role": "assistant", "content": content})
            else:
                raise ValueError(f"Failed to serialize message due to invalid role: {role}")

        return serialized_messages
