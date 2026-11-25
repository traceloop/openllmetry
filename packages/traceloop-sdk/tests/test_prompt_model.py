import importlib
import warnings

import traceloop.sdk.prompts.model as prompt_model


def test_prompt_model_import_emits_no_pydantic_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(prompt_model)

    deprecations = [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "class-based `config`" in str(w.message)
    ]
    assert deprecations == []
    assert prompt_model.RegistryObjectBaseModel.model_config.get(
        "arbitrary_types_allowed"
    )
