# Contributing to OpenLLMetry

We welcome contributions of all sizes to OpenLLMetry! Thank you for considering contributing to our project.

## Overview

It's the early days of our project, and we’re dedicated to building an awesome, inclusive community. To foster this environment, all community members must adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Community

Join our [Slack community](https://traceloopcommunity.slack.com/join/shared_invite/zt-1plpfpm6r-zOHKI028VkpcWdobX65C~g#/shared-invite/email) to chat and get help with any issues you may encounter.

### Bugs and Issues

Bug reports help make OpenLLMetry a better experience for everyone. When you report a bug, a template will be created automatically to gather necessary information.

Before raising a new issue, please search existing ones to avoid duplicates. If the issue is related to security, please email us directly at [dev@traceloop.com](mailto:dev@traceloop.com).

## Deciding What to Work On

You can start by browsing through our list of [issues](https://github.com/traceloop/openllmetry/issues) or adding your own to improve the test suite experience. Once you've decided on an issue, leave a comment and wait for approval to avoid multiple contributors working on the same task.

If you're unsure whether a proposed feature aligns with OpenLLMetry, feel free to raise an issue about it, and we'll respond promptly.

## Writing and Submitting Code

Anyone can contribute code to OpenLLMetry. To get started, check out the **Local Development Guide** below, make your changes, and submit a pull request to the main repository.

## Licensing

All of OpenLLMetry’s code is under the Apache 2.0 license. Any third-party components incorporated into our code are licensed under the original license provided by the respective component owner.

---

## Local Development

You can contribute by adding new instrumentations or updating and improving the different SDKs.

### Environment Setup

The Python and TypeScript projects are monorepos that use `nx` to manage the different packages. Ensure you have Node.js version 18 or higher and `nx` installed globally.

#### Basic Guide for Using `nx`

Most commands can be run from the root of the project. For example, to lint the entire project, run:

```bash
nx run-many -t lint
```
You can also use other commands similarly: `test`, `build`, `lock`, and `install` (for Python).

To run a specific command on a specific package, use:

```bash
nx run <package>:<command>
```
### Python Development
We use `poetry` to manage packages, and each package is managed independently under its own directory in `/packages`. All instrumentations depend on `opentelemetry-semantic-conventions-ai`, and `traceloop-sdk` depends on all the instrumentations.

When adding a new instrumentation, make sure to use it in `traceloop-sdk` and write proper tests.

### Debugging
Regardless of whether you’re working on an instrumentation or the SDK, we recommend testing your changes using the SDK in the sample app located in `/packages/sample-app` or running tests under the SDK.

Running Tests
We record HTTP requests and then replay them in tests to avoid making actual calls to the foundation model providers. See `vcr.py` and `pollyjs` for more details on usage and re-recording requests.

To run all tests, execute:
```bash
nx run-many -t test
```
To run a specific test, use:
```bash

nx run <package>:test
```
For example, to run the tests for the OpenAI instrumentation package, run:

```bash
nx run opentelemetry-instrumentation-openai:test
```
### TypeScript Development
We use `npm` with workspaces to manage packages in the monorepo. Install dependencies by running:

```bash
npm install
```
Each package has its own test suite. You can use the sample app to run and test changes locally.

Thank you for your interest in contributing to OpenLLMetry! We look forward to collaborating with you!



