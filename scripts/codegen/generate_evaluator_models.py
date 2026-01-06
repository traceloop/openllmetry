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
