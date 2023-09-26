import logging
import os
import threading
import time
import typing
import requests

from threading import Thread, Event
from typing import Optional
from jinja2 import Environment, meta
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential, retry_if_exception

from traceloop.sdk.prompts.model import Prompt, PromptVersion, TemplateEngine
from traceloop.sdk.prompts.registry import PromptRegistry

MAX_RETRIES = os.getenv("TRACELOOP_PROMPT_MANAGER_MAX_RETRIES") or 3
POLLING_INTERVAL = os.getenv("TRACELOOP_PROMPT_MANAGER_POLLING_INTERVAL") or 5
PROMPTS_ENDPOINT = "https://app.traceloop.com/api/prompts"


def get_effective_version(prompt: Prompt) -> PromptVersion:
    if len(prompt.versions) == 0:
        raise Exception(f"No versions exist for {prompt.key} prompt")

    return next(v for v in prompt.versions if v.id == prompt.target.version)


class PromptRegistryClient:
    _poller_thread: Thread
    _exit_monitor: Thread
    _registry: PromptRegistry
    _jinja_env: Environment
    _stop_polling_thread: Event

    def __new__(cls) -> "PromptRegistryClient":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(PromptRegistryClient, cls).__new__(cls)
            obj._registry = PromptRegistry()
            obj._jinja_env = Environment()
            obj._stop_polling_event = Event()
            obj._exit_monitor = Thread(target=monitor_exit, args=(obj._stop_polling_event,), daemon=True)
            obj._poller_thread = Thread(target=refresh_prompts, args=(
                obj._registry, obj._stop_polling_event, POLLING_INTERVAL))

        return cls.instance

    def run(self):
        response = fetch_url(PROMPTS_ENDPOINT)
        self._registry.load(response)
        self._exit_monitor.start()
        self._poller_thread.start()

    def render_prompt(self, key: str, **args):
        prompt = self._registry.get_prompt_by_key(key)
        if prompt is None:
            raise Exception(f"Prompt {key} does not exist")

        prompt_version = get_effective_version(prompt)
        params_dict = {
            "messages": self.render_messages(prompt_version, **args)
        }
        params_dict.update(prompt_version.llm_config)
        params_dict.pop("mode")

        return params_dict

    def render_messages(self, prompt_version: PromptVersion, **args):
        if prompt_version.templating_engine == TemplateEngine.JINJA2:
            rendered_messages = []
            for msg in prompt_version.messages:
                template = self._jinja_env.from_string(msg.template)
                template_variables = meta.find_undeclared_variables(self._jinja_env.parse(msg.template))
                missing_variables = template_variables.difference(set(args.keys()))
                if missing_variables == set():
                    rendered_msg = template.render(args)
                else:
                    raise Exception(f"Input variables: {','.join(str(var) for var in missing_variables)} are missing")

                # TODO: support other types than openai chat structure
                rendered_messages.append({"role": msg.role, "content": rendered_msg})

            return rendered_messages
        else:
            raise Exception(f"Templating engine {prompt_version.templating_engine} is not supported")


class RetryIfServerError(retry_if_exception):
    def __init__(
            self,
            exception_types: typing.Union[
                typing.Type[BaseException],
                typing.Tuple[typing.Type[BaseException], ...],
            ] = Exception,
    ) -> None:
        self.exception_types = exception_types
        super().__init__(lambda e: check_http_error(e))


def check_http_error(e):
    return isinstance(e, requests.exceptions.HTTPError) and (500 <= e.response.status_code < 600)


@retry(
    wait=wait_exponential(multiplier=1, min=4),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=RetryIfServerError(),
)
def fetch_url(url):
    response = requests.get(url, headers={
        "Authorization": f"Bearer {os.getenv('TRACELOOP_API_KEY')}"
    })

    if response.status_code != 200:
        raise requests.exceptions.HTTPError(response=response)
    else:
        return response.json()


def refresh_prompts(
        prompt_registry: PromptRegistry,
        stop_polling_event: Event,
        seconds_interval: Optional[int] = 5,
        endpoint: Optional[str] = PROMPTS_ENDPOINT
):
    while not stop_polling_event.is_set():
        try:
            response = fetch_url(endpoint)
            prompt_registry.load(response)
        except RetryError:
            logging.error("Request failed after retries : stopped polling")
            break

        time.sleep(seconds_interval)


def monitor_exit(exit_event: Event):
    main_thread = threading.main_thread()
    main_thread.join()
    exit_event.set()
