#!/bin/bash
set -euo pipefail

# Generate Pydantic models from OpenAPI/Swagger spec
# Extracts models used by v2/evaluators/execute/* endpoints
# Usage: ./scripts/generate-models.sh /path/to/swagger.json

if [ $# -eq 0 ]; then
    echo "Usage: $0 <path-to-swagger.json>"
    echo "Example: $0 /path/to/api-service/docs/swagger.json"
    exit 1
fi

SWAGGER_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/packages/traceloop-sdk/traceloop/sdk/generated/evaluators"
CODEGEN_SCRIPT="${SCRIPT_DIR}/codegen/generate_evaluator_models.py"

if [ ! -f "${SWAGGER_PATH}" ]; then
    echo "Error: Swagger file not found at ${SWAGGER_PATH}"
    exit 1
fi

echo "=== Generating models from ${SWAGGER_PATH} ==="

# Change to traceloop-sdk directory for uv
cd "${ROOT_DIR}/packages/traceloop-sdk"

# Run the Python generation script
uv run python "${CODEGEN_SCRIPT}" "${SWAGGER_PATH}" "${OUTPUT_DIR}"

echo ""
echo "Generated files:"
ls -la "${OUTPUT_DIR}"
