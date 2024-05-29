# Manages list of associated properties for which content tracing
# (prompts, vector embeddings, etc.) is allowed or disallowed.
class ContentAllowBlockList:
    def __new__(cls) -> "ContentAllowBlockList":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(ContentAllowBlockList, cls).__new__(cls)
            obj._allow_list = []
            obj._block_list = []

        return cls.instance

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

    def is_blocked(self, association_properties: dict) -> bool:
        for block_list_item in self._block_list:
            if any(
                [
                    association_properties.get(key) == value
                    for key, value in block_list_item.items()
                ]
            ):
                return True

        return False

    def load(self, response_allow_json: dict, response_block_json: dict):
        self._allow_list = response_allow_json["associationPropertyAllowList"]
        self._block_list = response_block_json["associationPropertyBlockList"]
