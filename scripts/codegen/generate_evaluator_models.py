#!/usr/bin/env python3
"""
Generate Pydantic models from OpenAPI/Swagger spec.
Extracts models used by v2/evaluators/execute/* endpoints.

Usage:
    python generate_evaluator_models.py <swagger_path> <output_dir>
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_definitions_and_mappings(swagger_path: str) -> tuple[dict, dict]:
    """
    Extract definitions used by v2/evaluators/execute/* endpoints.
    Also extracts slug-to-model mappings.

    Returns:
        tuple: (filtered_definitions, slug_mappings)
        slug_mappings: {slug: {"request": "ModelName", "response": "ModelName"}}
    """
    with open(swagger_path) as f:
        data = json.load(f)

    all_definitions = data["definitions"]
    needed_refs = set()
    slug_mappings = {}

    # Collect all definitions referenced by target endpoints
    for path, methods in data["paths"].items():
        if "/v2/evaluators/execute/" in path:
            # Extract slug from path like /v2/evaluators/execute/pii-detector
            slug = path.split("/v2/evaluators/execute/")[-1]
            if slug:
                slug_mappings[slug] = {"request": None, "response": None}

            for method, details in methods.items():
                # Get request body refs
                for param in details.get("parameters", []):
                    if "schema" in param and "$ref" in param["schema"]:
                        ref = param["schema"]["$ref"].replace("#/definitions/", "")
                        needed_refs.add(ref)
                        if slug and ref.startswith("request."):
                            # Convert request.PIIDetectorRequest to PIIDetectorRequest
                            model_name = ref.split(".")[-1]
                            slug_mappings[slug]["request"] = model_name

                # Get response refs
                for code, resp in details.get("responses", {}).items():
                    if "schema" in resp and "$ref" in resp["schema"]:
                        ref = resp["schema"]["$ref"].replace("#/definitions/", "")
                        needed_refs.add(ref)
                        # Only use 200 response for the success model mapping
                        if slug and code == "200" and ref.startswith("response."):
                            model_name = ref.split(".")[-1]
                            slug_mappings[slug]["response"] = model_name

    # Recursively find all referenced definitions
    def find_refs(obj, refs):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"].replace("#/definitions/", "")
                if ref not in refs:
                    refs.add(ref)
                    if ref in all_definitions:
                        find_refs(all_definitions[ref], refs)
            else:
                for v in obj.values():
                    find_refs(v, refs)
        elif isinstance(obj, list):
            for item in obj:
                find_refs(item, refs)

    # Find all nested references
    all_needed = set(needed_refs)
    for ref in list(needed_refs):
        if ref in all_definitions:
            find_refs(all_definitions[ref], all_needed)

    # Filter definitions to only include needed ones
    filtered_definitions = {
        k: v for k, v in all_definitions.items() if k in all_needed
    }

    # Clean up slug_mappings - remove entries without both request and response
    slug_mappings = {
        slug: models
        for slug, models in slug_mappings.items()
        if models["request"] and models["response"]
    }

    return filtered_definitions, slug_mappings


def generate_registry_py(output_dir: Path, slug_mappings: dict) -> int:
    """Generate registry.py with slug-to-model mappings."""

    # Collect all unique request and response model names
    request_models = sorted(set(
        m["request"] for m in slug_mappings.values() if m["request"]
    ))
    response_models = sorted(set(
        m["response"] for m in slug_mappings.values() if m["response"]
    ))

    content = '''"""
Registry mapping evaluator slugs to their request/response Pydantic models.

This enables type-safe validation of inputs and parsing of outputs.

DO NOT EDIT MANUALLY - Regenerate with:
    ./scripts/generate-models.sh /path/to/swagger.json
"""

from typing import Dict, Type, Optional
from pydantic import BaseModel

'''

    # Import request models
    if request_models:
        content += "from .request import (\n"
        for model in request_models:
            content += f"    {model},\n"
        content += ")\n\n"

    # Import response models
    if response_models:
        content += "from .response import (\n"
        for model in response_models:
            content += f"    {model},\n"
        content += ")\n\n"

    # Generate REQUEST_MODELS dict
    content += "\n# Mapping from evaluator slug to request model\n"
    content += "REQUEST_MODELS: Dict[str, Type[BaseModel]] = {\n"
    for slug in sorted(slug_mappings.keys()):
        model = slug_mappings[slug]["request"]
        if model:
            content += f'    "{slug}": {model},\n'
    content += "}\n\n"

    # Generate RESPONSE_MODELS dict
    content += "# Mapping from evaluator slug to response model\n"
    content += "RESPONSE_MODELS: Dict[str, Type[BaseModel]] = {\n"
    for slug in sorted(slug_mappings.keys()):
        model = slug_mappings[slug]["response"]
        if model:
            content += f'    "{slug}": {model},\n'
    content += "}\n\n"

    # Add helper functions
    content += '''
def get_request_model(slug: str) -> Optional[Type[BaseModel]]:
    """Get the request model for an evaluator by slug."""
    return REQUEST_MODELS.get(slug)


def get_response_model(slug: str) -> Optional[Type[BaseModel]]:
    """Get the response model for an evaluator by slug."""
    return RESPONSE_MODELS.get(slug)
'''

    (output_dir / "registry.py").write_text(content)

    return len(slug_mappings)


def generate_init_py(output_dir: Path) -> tuple[int, int]:
    """Generate __init__.py with proper exports."""
    # Extract class names from request.py
    request_classes = []
    request_file = output_dir / "request.py"
    if request_file.exists():
        content = request_file.read_text()
        request_classes = re.findall(r"^class (\w+)\(", content, re.MULTILINE)

    # Extract class names from response.py
    response_classes = []
    response_file = output_dir / "response.py"
    if response_file.exists():
        content = response_file.read_text()
        response_classes = re.findall(r"^class (\w+)\(", content, re.MULTILINE)

    # Generate __init__.py
    init_content = '''# generated by datamodel-codegen
# Models for v2/evaluators/execute endpoints from OpenAPI spec
#
# DO NOT EDIT MANUALLY - Regenerate with:
#   ./scripts/generate-models.sh /path/to/swagger.json

'''

    # Import request models
    if request_classes:
        init_content += "from .request import (\n"
        for cls in sorted(request_classes):
            init_content += f"    {cls},\n"
        init_content += ")\n\n"

    # Import factory class
    init_content += "from .factories import EvaluatorMadeByTraceloop\n\n"

    # Import registry
    init_content += """from .registry import (
    REQUEST_MODELS,
    RESPONSE_MODELS,
    get_request_model,
    get_response_model,
)

"""

    # Import response models
    if response_classes:
        init_content += "from .response import (\n"
        for cls in sorted(response_classes):
            init_content += f"    {cls},\n"
        init_content += ")\n\n"

    # Generate __all__
    init_content += "__all__ = [\n"
    init_content += "    # Factory class\n"
    init_content += '    "EvaluatorMadeByTraceloop",\n'
    init_content += "    # Registry functions\n"
    init_content += '    "REQUEST_MODELS",\n'
    init_content += '    "RESPONSE_MODELS",\n'
    init_content += '    "get_request_model",\n'
    init_content += '    "get_response_model",\n'

    if request_classes:
        init_content += "    # Evaluator request models\n"
        for cls in sorted(request_classes):
            init_content += f'    "{cls}",\n'
    if response_classes:
        init_content += "    # Evaluator response models\n"
        for cls in sorted(response_classes):
            init_content += f'    "{cls}",\n'
    init_content += "]\n"

    (output_dir / "__init__.py").write_text(init_content)

    return len(request_classes), len(response_classes)


def generate_factories_py(
    output_dir: Path, slug_mappings: dict, filtered_definitions: dict
) -> int:
    """Generate factories.py with static factory methods for each evaluator.

    This creates type-safe factory methods with proper IDE autocomplete support.
    """

    def slug_to_method_name(slug: str) -> str:
        """Convert slug like 'pii-detector' to method name like 'pii_detector'."""
        return slug.replace("-", "_")

    def get_type_hint(prop: dict) -> str:
        """Convert JSON schema property to Python type hint."""
        prop_type = prop.get("type")
        if prop_type == "number":
            return "float"
        elif prop_type == "integer":
            return "int"
        elif prop_type == "boolean":
            return "bool"
        elif prop_type == "string":
            return "str"
        elif prop_type == "array":
            items = prop.get("items", {})
            item_type = get_type_hint(items) if items else "Any"
            return f"list[{item_type}]"
        return "Any"

    def get_config_fields(request_model_name: str) -> list[dict]:
        """Extract config fields from the request model's config property."""
        # Find the request model definition
        request_def_key = f"request.{request_model_name}"
        if request_def_key not in filtered_definitions:
            return []

        request_def = filtered_definitions[request_def_key]
        properties = request_def.get("properties", {})

        # Check if there's a config property
        if "config" not in properties:
            return []

        config_prop = properties["config"]
        # Get the config model reference
        if "$ref" in config_prop:
            config_ref = config_prop["$ref"].replace("#/definitions/", "")
            if config_ref in filtered_definitions:
                config_def = filtered_definitions[config_ref]
                config_props = config_def.get("properties", {})
                config_required = set(config_def.get("required", []))

                fields = []
                for name, prop in config_props.items():
                    field_type = get_type_hint(prop)
                    is_required = name in config_required
                    examples = prop.get("examples", [])
                    example = examples[0] if examples else None
                    fields.append({
                        "name": name,
                        "type": field_type,
                        "required": is_required,
                        "example": example,
                    })
                return fields
        return []

    def get_input_fields(request_model_name: str) -> list[str]:
        """Extract required input fields from the request model's input property."""
        request_def_key = f"request.{request_model_name}"
        if request_def_key not in filtered_definitions:
            return []

        request_def = filtered_definitions[request_def_key]
        properties = request_def.get("properties", {})

        # Check if there's an input property
        if "input" not in properties:
            return []

        input_prop = properties["input"]
        if "$ref" in input_prop:
            input_ref = input_prop["$ref"].replace("#/definitions/", "")
            if input_ref in filtered_definitions:
                input_def = filtered_definitions[input_ref]
                # Return all required fields from the input model
                return list(input_def.get("required", []))
        return []

    # Start building the factories.py content
    content = '''"""
Factory methods for creating Traceloop evaluators.

Provides type-safe factory methods with IDE autocomplete support.

DO NOT EDIT MANUALLY - Regenerate with:
    ./scripts/generate-models.sh /path/to/swagger.json
"""
from __future__ import annotations

from ...evaluator.config import EvaluatorDetails


class EvaluatorMadeByTraceloop:
    """
    Factory class for creating Traceloop evaluators with type-safe configuration.

    Each method creates an EvaluatorDetails instance for a specific evaluator,
    with properly typed configuration parameters.

    Example:
        >>> from traceloop.sdk.evaluator import EvaluatorMadeByTraceloop
        >>>
        >>> evaluators = [
        ...     EvaluatorMadeByTraceloop.pii_detector(probability_threshold=0.8),
        ...     EvaluatorMadeByTraceloop.toxicity_detector(threshold=0.7),
        ...     EvaluatorMadeByTraceloop.faithfulness(),
        ... ]
    """

'''

    # Generate a method for each evaluator
    for slug in sorted(slug_mappings.keys()):
        request_model = slug_mappings[slug]["request"]
        method_name = slug_to_method_name(slug)
        config_fields = get_config_fields(request_model)
        input_fields = get_input_fields(request_model)

        # Build method signature
        params = []
        for field in config_fields:
            type_hint = field["type"]
            if not field["required"]:
                type_hint = f"{type_hint} | None"
            default = " = None" if not field["required"] else ""
            params.append(f"{field['name']}: {type_hint}{default}")

        # Method signature
        if params:
            params_str = ",\n        ".join(params)
            content += "    @staticmethod\n"
            content += f"    def {method_name}(\n"
            content += f"        {params_str},\n"
            content += "    ) -> EvaluatorDetails:\n"
        else:
            content += "    @staticmethod\n"
            content += f"    def {method_name}() -> EvaluatorDetails:\n"

        # Docstring
        content += f'        """Create {slug} evaluator.\n'
        if config_fields:
            content += "\n        Args:\n"
            for field in config_fields:
                example_str = f" (example: {field['example']})" if field["example"] is not None else ""
                content += f"            {field['name']}: {field['type']}{example_str}\n"
        if input_fields:
            content += f"\n        Required input fields: {', '.join(input_fields)}\n"
        content += '        """\n'

        # Method body
        if config_fields:
            # Build config dict, filtering out None values
            config_items = ", ".join(
                f'"{f["name"]}": {f["name"]}' for f in config_fields
            )
            content += "        config = {\n"
            content += f"            k: v for k, v in {{{config_items}}}.items()\n"
            content += "            if v is not None\n"
            content += "        }\n"
            content += "        return EvaluatorDetails(\n"
            content += f'            slug="{slug}",\n'
            content += "            config=config if config else None,\n"
            if input_fields:
                content += f"            required_input_fields={input_fields},\n"
            content += "        )\n\n"
        else:
            content += "        return EvaluatorDetails(\n"
            content += f'            slug="{slug}",\n'
            if input_fields:
                content += f"            required_input_fields={input_fields},\n"
            content += "        )\n\n"

    # Remove trailing whitespace to pass lint
    content = content.rstrip() + "\n"

    (output_dir / "factories.py").write_text(content)

    return len(slug_mappings)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <swagger_path> <output_dir>")
        print(f"Example: {sys.argv[0]} /path/to/swagger.json ./generated")
        sys.exit(1)

    swagger_path = sys.argv[1]
    output_dir = Path(sys.argv[2])

    if not Path(swagger_path).exists():
        print(f"Error: Swagger file not found at {swagger_path}")
        sys.exit(1)

    print("=== Extracting definitions for evaluator execute endpoints ===")

    # Extract definitions and slug mappings
    filtered_definitions, slug_mappings = extract_definitions_and_mappings(
        swagger_path
    )

    request_count = len(
        [k for k in filtered_definitions if k.startswith("request.")]
    )
    response_count = len(
        [k for k in filtered_definitions if k.startswith("response.")]
    )

    print(f"Extracted {len(filtered_definitions)} definitions")
    print(f"Request types: {request_count}")
    print(f"Response types: {response_count}")
    print(f"Evaluator slugs: {len(slug_mappings)}")

    # Create JSON Schema with filtered definitions
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": filtered_definitions,
        "type": "object",
    }

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(schema, f, indent=2)
        temp_schema = f.name

    print("=== Generating Pydantic models ===")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run datamodel-codegen
    try:
        subprocess.run(
            [
                "datamodel-codegen",
                "--input", temp_schema,
                "--input-file-type", "jsonschema",
                "--output", str(output_dir),
                "--output-model-type", "pydantic_v2.BaseModel",
                "--target-python-version", "3.10",
                "--use-standard-collections",
                "--reuse-model",
                "--disable-timestamp",
            ],
            check=True,
        )
    finally:
        # Cleanup temp file
        Path(temp_schema).unlink(missing_ok=True)

    print("=== Generating registry.py with slug mappings ===")
    registry_count = generate_registry_py(output_dir, slug_mappings)
    print(f"Generated registry.py with {registry_count} evaluator mappings")

    print("=== Generating factories.py with evaluator factory methods ===")
    factory_count = generate_factories_py(output_dir, slug_mappings, filtered_definitions)
    print(f"Generated factories.py with {factory_count} evaluator factory methods")

    print("=== Generating __init__.py with exports ===")
    req_count, resp_count = generate_init_py(output_dir)
    print(
        f"Generated __init__.py with {req_count} request "
        f"and {resp_count} response exports"
    )

    print("=== Model generation complete ===")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
