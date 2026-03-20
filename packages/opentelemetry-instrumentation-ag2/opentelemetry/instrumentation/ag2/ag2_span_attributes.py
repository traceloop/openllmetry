from opentelemetry.trace import Span


def set_span_attribute(span: Span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


class AG2SpanAttributes:
    def __init__(self, span: Span, instance) -> None:
        self.span = span
        self.instance = instance
        self.process_instance()

    def process_instance(self):
        instance_type = self.instance.__class__.__name__
        method_mapping = {
            "ConversableAgent": self._process_agent,
            "AssistantAgent": self._process_agent,
            "UserProxyAgent": self._process_agent,
            "GroupChatManager": self._process_group_chat_manager,
        }
        method = method_mapping.get(instance_type)
        if method:
            method()

    def _process_agent(self):
        if hasattr(self.instance, "name"):
            self._set_attribute("gen_ai.agent.name", self.instance.name)
        if hasattr(self.instance, "description"):
            self._set_attribute("gen_ai.agent.description", self.instance.description)
        if hasattr(self.instance, "_oai_system_message") and self.instance._oai_system_message:
            msgs = self.instance._oai_system_message
            system_msg = msgs[0].get("content", "") if msgs else ""
            self._set_attribute("ag2.agent.system_message", system_msg)

    def _process_group_chat_manager(self):
        if hasattr(self.instance, "name"):
            self._set_attribute("gen_ai.agent.name", self.instance.name)
        if hasattr(self.instance, "groupchat"):
            groupchat = self.instance.groupchat
            if hasattr(groupchat, "agents"):
                agent_names = [a.name for a in groupchat.agents if hasattr(a, "name")]
                self._set_attribute("ag2.group_chat.agents", str(agent_names))
            if hasattr(groupchat, "max_round"):
                self._set_attribute("ag2.group_chat.max_round", groupchat.max_round)

    def _set_attribute(self, key, value):
        if value is not None:
            set_span_attribute(self.span, key, str(value) if isinstance(value, list) else value)
