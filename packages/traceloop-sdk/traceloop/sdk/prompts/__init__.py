from traceloop.sdk import PromptRegistryClient


def render_by_key(key, **args):
    PromptRegistryClient().render_prompt(key, **args)
