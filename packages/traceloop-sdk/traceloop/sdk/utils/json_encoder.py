import dataclasses
import inspect
import json


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, dict):
            if "callbacks" in o:
                del o["callbacks"]
                return o

        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        to_json_method = getattr(o, "to_json", None)

        if callable(to_json_method):
            return to_json_method()

        json_method = getattr(o, "json", None)

        if callable(json_method):
            try:
                result = json_method()

                if inspect.iscoroutine(result):
                    result.close()
                    return o.__class__.__name__

                return result

            except Exception:
                pass

        if hasattr(o, "__class__"):
            return o.__class__.__name__

        return super().default(o)