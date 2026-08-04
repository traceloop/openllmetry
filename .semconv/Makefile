# Vendors the upstream GenAI semantic conventions and resolves them to JSON.
# All Docker invocation lives here. Nothing else in the repo runs weaver.
include versions.env

SELF_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
BUILD    := .build
REGISTRY := registry
UPSTREAM := $(BUILD)/sc-upstream-$(SEMCONV_VERSION)
FILTERED := $(BUILD)/sc-upstream-filtered

WEAVER := docker run --rm -u $(shell id -u):$(shell id -g) \
	-v "$(SELF_DIR):/w" -w /w -e HOME=/tmp otel/weaver:$(WEAVER_VERSION)

.PHONY: vendor resolve generate check verify-vendor clean

## Fetch the pinned registry into registry/ .
vendor:
	rm -rf $(REGISTRY) $(BUILD)
	mkdir -p $(BUILD)
	git clone -q $(SEMCONV_GENAI_REPO) $(BUILD)/genai
	cd $(BUILD)/genai && git checkout -q $(SEMCONV_GENAI_REF)
	cp -r $(BUILD)/genai/model $(REGISTRY)
	rm -rf $(BUILD)/genai
	@echo "vendored $(SEMCONV_GENAI_REF) -> $(REGISTRY)/"

## Prove registry/ still matches the pinned upstream SHA. `check` only proves
## generated.py matches registry/ -- it never proves registry/ itself hasn't been
## hand-edited. This clones the pin fresh into a scratch dir and diffs it against
## the committed registry/ tree, so a weakened vendored spec can't sneak through
## alongside a regenerated artifact.
verify-vendor:
	rm -rf $(BUILD)/verify-vendor
	mkdir -p $(BUILD)/verify-vendor
	git clone -q $(SEMCONV_GENAI_REPO) $(BUILD)/verify-vendor/genai
	cd $(BUILD)/verify-vendor/genai && git checkout -q $(SEMCONV_GENAI_REF)
	@if ! diff -r --exclude=.build $(BUILD)/verify-vendor/genai/model $(REGISTRY); then \
		echo ""; \
		echo "registry/ does not match upstream $(SEMCONV_GENAI_REF) -- run 'make -C .semconv vendor' to refresh it"; \
		rm -rf $(BUILD)/verify-vendor; \
		exit 1; \
	fi
	rm -rf $(BUILD)/verify-vendor
	@echo "registry/ matches upstream $(SEMCONV_GENAI_REF)"

## Build the filtered upstream dependency the registry manifest points at.
$(FILTERED):
	mkdir -p $(BUILD)
	git clone -q --depth 1 --branch $(SEMCONV_VERSION) \
		https://github.com/open-telemetry/semantic-conventions.git $(UPSTREAM)
	mkdir -p $(dir $(FILTERED))
	cp -r $(UPSTREAM)/model $(FILTERED)
	cd $(FILTERED) && rm -rf gen-ai mcp openai

## Resolve the registry to a single JSON document.
resolve: $(FILTERED)
	$(WEAVER) registry resolve -r $(REGISTRY) --format json -o /w/$(BUILD)/resolved.json
	@echo "resolved -> $(BUILD)/resolved.json"

## Regenerate the committed contract module.
generate: resolve
	cd ../packages/opentelemetry-semantic-conventions-ai && uv run python -m \
		opentelemetry.semconv_ai._contract._generator \
		--resolved ../../.semconv/$(BUILD)/resolved.json \
		--out opentelemetry/semconv_ai/_contract/generated.py

## Fail if the committed artifact is stale.
check: generate
	cd .. && git diff --exit-code -- \
		packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/_contract/generated.py

clean:
	rm -rf $(BUILD) $(REGISTRY)/.build
