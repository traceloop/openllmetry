# Contributing to OpenLLMetry

Thanks for taking the time to contribute! 😃 🚀

Please refer to our [Contributing Guide](https://traceloop.com/docs/openllmetry/contributing/overview) for instructions on how to contribute.

## Local Testing and Linting in this Repo

A few steps to set up this repo locally.

Run the following at repo root to setup the yarn dependencies.
```shell
npm ci
```

Make sure `uv` is installed for python packages managed by `uv`.

On Linux, set `export UV_LINK_MODE=copy` before installing. uv hardlinks packages
from its cache there, and nltk>=3.10 (pulled in by llama-index, e.g. in the
llamaindex instrumentation and the sample app) refuses to read hardlinked data files.

Generally, for setting up and testing an individual package, run the following from repo root.

```shell
npx nx run opentelemetry-instrumentation-openai:install
npx nx run opentelemetry-instrumentation-openai:lint
npx nx run opentelemetry-instrumentation-openai:test
```

Or you can run the following to automatically set up all affected packages.
```shell
npx nx affected -t install
npx nx affected -t lint
npx nx affected -t test
```

At the package directory, you can run `nx` without specifying the package.
```shell
cd packages/opentelemetry-instrumentation-openai
npx nx install
npx nx lint
npx nx test
```
