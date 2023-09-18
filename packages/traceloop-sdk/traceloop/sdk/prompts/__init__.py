from traceloop.sdk.prompts.client import PromptRegistryClient


def render_by_key(key, **args):
    PromptRegistryClient().render_prompt(key, **args)
