# Contributing to OpenLLMetry

Thanks for taking the time to contribute! 😃 🚀

Please refer to our [Contributing Guide](https://traceloop.com/docs/openllmetry/contributing/overview) for instructions on how to contribute.

## Local Testing and Linting in this Repo

A few steps to set up this repo locally.

Run the following at repo root to setup the yarn dependencies.
```shell
npm ci
```

Install `poetry` (version 2) at the system-level (e.g. using `pipx` or `brew`).

`poetry` does not recognize the `.python-version` specification in python projects.
It has to be told explicitly what version to use.
Otherwise, it may attempt to use a newer, unsupported python version and encounter build errors.

```shell
cd packages/opentelemetry-instrumentation-openai/
poetry python install $(head -n1 .python-version)
poetry env use $(head -n1 .python-version)
poetry install
```

Generally, for setting up and testing an individual package, run the following from repo root.

```shell
npx nx run opentelemetry-instrumentation-openai:install --with dev,test
npx nx run opentelemetry-instrumentation-openai:lint
npx nx run opentelemetry-instrumentation-openai:test
```

Or you can run the following to automatically set up all affected packages.
```shell
npx nx affected -t install --with dev,test
npx nx affected -t lint
npx nx affected -t test
```
