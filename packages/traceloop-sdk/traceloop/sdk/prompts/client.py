import atexit
import logging
import os
import time
import typing
import requests

from threading import Thread, Event
from typing import Optional
from jinja2 import Environment
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential, retry_if_exception

from traceloop.sdk.prompts.model import Prompt, PromptVersion, TemplateEngine
from traceloop.sdk.prompts.registry import PromptRegistry

TRACELOOP_BASE_URL = os.getenv("TRACELOOP_BASE_URL") or "https://app-staging.traceloop.dev"
MAX_RETRIES = os.getenv("TRACELOOP_PROMPT_MANAGER_MAX_RETRIES") or 3
POLLING_INTERVAL = os.getenv("TRACELOOP_PROMPT_MANAGER_POLLING_INTERVAL") or 5
PROMPTS_ENDOINT = f"{TRACELOOP_BASE_URL}/api/prompts"


class PromptRegistryClient:
    _poller_thread: Thread
    _registry: PromptRegistry
    _jinja_env: Environment
    _stop_polling_thread: Event

    def __new__(cls) -> "PromptRegistryClient":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(PromptRegistryClient, cls).__new__(cls)
            obj._registry = PromptRegistry()
            obj._jinja_env = Environment()
            obj._stop_polling_event = Event()
            obj._poller_thread = Thread(target=refresh_prompts, args=(
                obj._registry, obj._stop_polling_event, POLLING_INTERVAL))

            atexit.register(obj._stop_polling_event.set)

        return cls.instance

    def run(self):
        response = fetch_url(PROMPTS_ENDOINT)
        self._registry.load(response)
        self._poller_thread.start()

    def render_prompt(self, key: str, **args):
        prompt = self._registry.get_prompt_by_key(key)
        if prompt is None:
            raise f"Prompt {key} does not exist"

        prompt_version = self.get_effective_version(prompt)
        params_dict = {
            "messages": self.render_messages(prompt_version, **args)
        }

        params_dict.update(prompt_version.model_config)

        return params_dict

    def render_messages(self, prompt_version: PromptVersion, **args):
        if prompt_version.templating_engine == TemplateEngine.JINJA2:
            rendered_messages = []
            for msg in prompt_version.messages:
                template = self._jinja_env.from_string(msg.template)
                rendered_msg = template.render(args)
                # TODO: support other types than openai chat structure
                rendered_messages.append({"role": msg.role, "content": rendered_msg})

            return rendered_messages
        else:
            raise f"Templating engine {prompt_version.templating_engine} is not supported"

    def get_effective_version(self, prompt: Prompt):
        if len(prompt.versions) == 0:
            raise f"No versions exist for {prompt.key} prompt"

        # TODO: get version by targeting
        return prompt.versions[0]


class RetryIfServerError(retry_if_exception):
    def __init__(
            self,
            exception_types: typing.Union[
                typing.Type[BaseException],
                typing.Tuple[typing.Type[BaseException], ...],
            ] = Exception,
    ) -> None:
        self.exception_types = exception_types
        super().__init__(lambda e: isinstance(e, requests.exceptions.HTTPError) and (500 <= e.code < 600))


@retry(
    wait=wait_exponential(multiplier=1, min=4),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=RetryIfServerError(),
)
def fetch_url(url):
    try:
        with requests.get(url) as response:
            return response.json()
    except requests.exceptions.HTTPError as e:
        raise e
    except Exception as e:
        if 500 <= e.code < 600:
            raise e


def refresh_prompts(
        prompt_registry: PromptRegistry,
        stop_polling_event: Event,
        seconds_interval: Optional[int] = 5,
        endpoint: Optional[str] = PROMPTS_ENDOINT
):
    while not stop_polling_event.is_set():
        try:
            response = fetch_url(endpoint)
            prompt_registry.load(response)
        except RetryError:
            logging.error("Request failed after retries : stopped polling")
            break

        time.sleep(seconds_interval)
