from traceloop.sdk.prompts.client import PromptRegistryClient


def render_prompt_by_key(key, **args):
    return PromptRegistryClient().render_prompt(key, **args)
