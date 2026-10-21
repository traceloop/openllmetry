import dspy
from opentelemetry.instrumentation.dspy import DSPyInstrumentor
from traceloop.sdk import Traceloop


class BasicQA(dspy.Signature):
    """Answer questions with short factual responses."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class QAPipeline(dspy.Module):
    def __init__(self):
        self.generate_answer = dspy.ChainOfThought(BasicQA)

    def forward(self, question):
        return self.generate_answer(question=question)


def main():
    Traceloop.init(app_name="dspy-example")
    DSPyInstrumentor().instrument()
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    pipeline = QAPipeline()
    result = pipeline(question="What is the capital of France?")
    print("Answer:", result.answer)


if __name__ == "__main__":
    main()
