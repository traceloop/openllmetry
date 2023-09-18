import json
import typing

from traceloop.sdk.prompts.model import Prompt


class PromptRegistry:
    def __init__(self):
        self._prompts: typing.Dict[str, Prompt] = dict()

    def get_prompt_by_key(self, key):
        return self._prompts.get(key, None)

    def loads(self, json_str):
        obj = json.loads(json_str)

        for prompt_obj in obj["prompts"]:
            self._prompts[prompt_obj["key"]] = Prompt(**prompt_obj)
