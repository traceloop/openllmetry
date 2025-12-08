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

For setting up and testing an individual package, run the following from repo root.

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

## Build Errors

If encountering local build errors in environment setup, try a lower python version, e.g. 3.11.
There are `.python-version` files in individual package directories, but poetry may not reliably respect it.
Depending on your python installation, you may need to set your python version accordingly at repo root (e.g. with a `.python-version` or `.tool-versions`).
The `.venv` dir within each package must be deleted before you retry the poetry/nx command, as it will not clear the existing venv and pick up the updated, lower version specified until the existing venv has been deleted manually.
