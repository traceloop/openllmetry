# Manages list of associated properties for which content tracing
# (prompts, vector embeddings, etc.) is allowed.
class ContentAllowList:
    def __init__(self):
        self._allow_list: list[dict] = []

    def is_allowed(self, association_properties: dict) -> bool:
        for allow_list_item in self._allow_list:
            if all(
                [
                    association_properties.get(key) == value
                    for key, value in allow_list_item.items()
                ]
            ):
                return True

        return False

    def load(self, response_json: dict):
        self._allow_list = response_json["associationPropertyAllowList"]
