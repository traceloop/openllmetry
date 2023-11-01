from jinja2 import Environment, meta
from traceloop.sdk.prompts.model import Prompt, PromptVersion, TemplateEngine
from traceloop.sdk.prompts.registry import PromptRegistry
from traceloop.sdk.tracing.tracing import set_prompt_tracing_context


def get_effective_version(prompt: Prompt) -> PromptVersion:
    if len(prompt.versions) == 0:
        raise Exception(f"No versions exist for {prompt.key} prompt")

    return next(v for v in prompt.versions if v.id == prompt.target.version)


class PromptRegistryClient:
    _registry: PromptRegistry
    _jinja_env: Environment

    def __new__(cls) -> "PromptRegistryClient":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(PromptRegistryClient, cls).__new__(cls)
            obj._registry = PromptRegistry()
            obj._jinja_env = Environment()

        return cls.instance

    def render_prompt(self, key: str, **args):
        prompt = self._registry.get_prompt_by_key(key)
        if prompt is None:
            raise Exception(f"Prompt {key} does not exist")

        prompt_version = get_effective_version(prompt)
        params_dict = {"messages": self.render_messages(prompt_version, **args)}
        params_dict.update(prompt_version.llm_config)
        params_dict.pop("mode")

        set_prompt_tracing_context(
            prompt.key, prompt_version.version, prompt_version.name, prompt_version.hash, args
        )

        return params_dict

    def render_messages(self, prompt_version: PromptVersion, **args):
        if prompt_version.templating_engine == TemplateEngine.JINJA2:
            rendered_messages = []
            for msg in prompt_version.messages:
                template = self._jinja_env.from_string(msg.template)
                template_variables = meta.find_undeclared_variables(
                    self._jinja_env.parse(msg.template)
                )
                missing_variables = template_variables.difference(set(args.keys()))
                if missing_variables == set():
                    rendered_msg = template.render(args)
                else:
                    raise Exception(
                        f"Input variables: {','.join(str(var) for var in missing_variables)} are missing"
                    )

                # TODO: support other types than openai chat structure
                rendered_messages.append({"role": msg.role, "content": rendered_msg})

            return rendered_messages
        else:
            raise Exception(
                f"Templating engine {prompt_version.templating_engine} is not supported"
            )
