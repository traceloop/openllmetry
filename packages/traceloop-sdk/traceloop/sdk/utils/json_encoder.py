import dataclasses
import json


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, dict):
            if "callbacks" in o:
                del o["callbacks"]
                return o
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if hasattr(o, "to_json"):
            return o.to_json()

        # Prefer Pydantic v2 model_dump() to avoid deprecation warnings.
        # We use model_dump()/dict() (returning dicts) instead of
        # model_dump_json()/json() (returning strings) to avoid
        # double-encoding in json.JSONEncoder.default().
        if hasattr(o, "model_dump"):
            return o.model_dump()

        if hasattr(o, "dict"):
            return o.dict()

        if hasattr(o, "__class__"):
            return o.__class__.__name__

        return super().default(o)
