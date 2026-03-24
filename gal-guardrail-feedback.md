# Gal's Guardrail API Feedback (PR #3649)

Source: https://github.com/traceloop/openllmetry/pull/3649#issuecomment-4062537175

Overall theme: The current API uses Java/C#-style static factory classes and is not Pythonic. Below are the 6 suggestions, each as a standalone item.

---

## 1. `OnFailure.*` — Replace static namespace class with strings / callables

**Current:**
```python
on_failure=OnFailure.raise_exception("PII detected in response")
on_failure=OnFailure.return_value(value="Sorry, I can't help you with that.")
on_failure=OnFailure.log(level=logging.WARNING, message="Guard failed")
on_failure=OnFailure.noop()
```

**Suggested — strings for common cases, callables for custom:**
```python
on_failure="raise"
on_failure="log"
on_failure="ignore"  # noop
on_failure="Sorry, I can't help you with that."  # string = return this value
on_failure=lambda result: log_and_alert(result)  # custom behavior
```

**Or — fluent chaining:**
```python
result = await guardrail.run(my_func).on_failure("raise")
result = await guardrail.run(my_func).on_failure(return_value="fallback")
```

**Rationale:** The `OnFailure` class is just 4 `@staticmethod` methods returning closures. Module-level functions or string literals achieve the same with less ceremony.

- [ ] Address this suggestion
- [ ] Test changes

---

## 2. `Condition.*` — Replace static namespace class with plain Python expressions

**Current:**
```python
condition=Condition.is_true()
condition=Condition.is_false()
condition=Condition.greater_than(0.8)
condition=Condition.less_than(1000)
condition=Condition.between(0.3, 0.7)
condition=Condition.equals("approved")
```

**Suggested — plain Python lambdas or short module-level helpers:**
```python
condition=bool
condition=lambda v: not v
condition=lambda v: v > 0.8
condition=lambda v: 0.3 <= v <= 0.7
condition=lambda v: v == "approved"
```

Or named helpers:
```python
from traceloop.sdk.guardrail.conditions import gt, lt, between, eq

condition=gt(0.8)
condition=between(0.3, 0.7)
```

**Rationale:** Every Python developer already knows `>`, `<`, `==`, `lambda`. Wrapping these in `Condition.greater_than_or_equal()` adds 30 characters for zero expressiveness.

- [ ] Address this suggestion
- [ ] Test changes

---

## 3. `client.create_guardrail(...)` — Replace factory method with direct construction

**Current:**
```python
client = Traceloop.init(app_name="my-app", ...)
guardrail = client.create_guardrail(
    guards=[pii_guard(), toxicity_guard()],
    on_failure=OnFailure.raise_exception("Content policy violation"),
    name="safety-check",
)
result = await guardrail.run(my_func)
```

**Suggested — direct construction:**
```python
guardrail = Guardrail(
    pii_guard(),
    toxicity_guard(),
    on_failure="raise",
    name="safety-check",
)
result = await guardrail.run(my_func)
```

**Or with chaining:**
```python
result = await Guardrail(pii_guard(), toxicity_guard()).run(my_func).on_failure("raise")
```

**Rationale:** If a class can be instantiated directly, it should be. The `client.create_guardrail()` indirection forces users to manage a client object just to create a guardrail. The HTTP client dependency can be resolved internally from the Traceloop singleton.

- [ ] Address this suggestion
- [ ] Test changes

---

## 4. `guardrail.run(lambda: func(...))` — Accept func + args directly instead of lambda wrapping

**Current:**
```python
return await g.run(
    lambda: func(*args, **kwargs),
    input_mapper=input_mapper,
)

result = await guardrail.run(generate_summary)
```

**Suggested — accept func + args directly:**
```python
result = await guardrail.run(generate_summary, prompt, model="gpt-4")
```

**Or as a transparent decorator:**
```python
@guardrail(pii_guard(), on_failure="raise")
async def generate_summary(prompt: str) -> str:
    return await openai.chat(prompt)

result = await generate_summary("tell me about X")
```

**Rationale:** This is how `asyncio.create_task(coro)`, `functools.partial(fn, *args)`, and `concurrent.futures.submit(fn, *args)` work in the stdlib. Python devs expect `run(fn, *args, **kwargs)`, not `run(lambda: fn(*args, **kwargs))`.

- [ ] Address this suggestion
- [ ] Test changes

---

## 5. `input_mapper` positional list — Replace with dict keyed by guard name

**Current:**
```python
input_mapper=lambda response: [
    AnswerRelevancyInput(answer=response, question=prompt),
    SexismDetectorInput(text=response),
    ToxicityDetectorInput(text=response),
]
```

The list elements are position-matched to the guards list. If guards are reordered, the mapper silently sends wrong inputs to wrong guards.

**Suggested — dict keyed by guard name:**
```python
input_mapper=lambda response: {
    "answer_relevancy": AnswerRelevancyInput(answer=response, question=prompt),
    "sexism": SexismDetectorInput(text=response),
    "toxicity": ToxicityDetectorInput(text=response),
}
```

**Or let each guard declare its own input mapping:**
```python
pii_guard(input=lambda r: PIIDetectorInput(text=r))
toxicity_guard(input=lambda r: ToxicityDetectorInput(text=r))
```

**Rationale:** Self-documenting and order-independent.

- [ ] Address this suggestion
- [ ] Test changes

---

## 6. Guard factory functions — Compose more naturally with sensible defaults

**Current:**
```python
guards=[
    toxicity_guard(condition=Condition.is_true()),
    pii_guard(condition=Condition.is_false()),
    custom_evaluator_guard(
        evaluator_slug="medicaladvice",
        condition=Condition.greater_than_or_equal(0.8),
    ),
]
```

**Suggested — more composable, reads like English:**
```python
guardrail = Guardrail(
    toxicity_guard(),           # sensible defaults, no Condition needed
    no_pii(),                   # name implies the condition
    custom_evaluator("medicaladvice", threshold=0.8),
    on_failure="raise",
)
```

**Rationale:** Guard names like `pii_guard` + `Condition.is_false()` don't encode their polarity. Renaming to `no_pii()` or `pii_free()` eliminates the need for an explicit condition in the common case.

- [ ] Address this suggestion
- [ ] Test changes
