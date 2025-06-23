"""
Simple Guardrails Example

This example demonstrates the basic usage of Traceloop's guardrails feature
for content validation and safety checks.
"""

import os
from openai import OpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.guardrails import GuardrailInputData, GuardrailConfig

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Traceloop.init(app_name="simple_guardrails_demo")

guardrails = Traceloop.get().guardrails


def generate_text(prompt: str) -> str:
    """Generate text using OpenAI"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content or ""


def validate_and_generate(user_input: str) -> str:
    """Validate user input with guardrails before generating text"""
    
    print(f"📝 User input: {user_input}")
    
    input_data = GuardrailInputData(
        content=user_input,
        context={"user_type": "demo_user"},
        metadata={"timestamp": "2024-01-01T10:00:00Z"}
    )
    
    config = GuardrailConfig(
        thresholds={"safety": 0.8},
        parameters={"check_categories": ["safety", "appropriateness"]}
    )
    
    try:
        result = guardrails.validate_input_sync(
            evaluator_slug="content-safety",
            input_data=input_data,
            config=config,
            timeout=10
        )
        
        print(f"🛡️ Validation result: {result.action.value}")
        
        if result.score:
            print(f"📊 Safety score: {result.score}")
        
        if result.pass_through:
            print("✅ Input is safe, generating response...")
            generated_text = generate_text(user_input)
            print(f"🤖 Generated: {generated_text}")
            return generated_text
        
        elif result.blocked:
            warning_msg = f"❌ Input blocked: {result.reason}"
            print(warning_msg)
            return warning_msg
        
        elif result.retry_required:
            warning_msg = f"⚠️ Input needs modification: {result.reason}"
            print(warning_msg)
            return warning_msg
        
        else:
            return "Unknown validation result"
            
    except Exception as e:
        error_msg = f"⚠️ Validation error: {str(e)}"
        print(error_msg)
        # In case of validation error, you can choose to:
        # 1. Block the request (safer)
        # 2. Allow it to proceed (less safe but more robust)
        # Here we'll be conservative and block
        return error_msg


def main():
    """Run the simple guardrails demo"""
    print("🚀 Simple Guardrails Demo")
    print("=" * 40)
    
    test_inputs = [
        "Tell me a joke about programming",
        "Write a poem about nature",
        "How do I bake a chocolate cake?",
        "Explain machine learning in simple terms",
        "What's the weather like today?"
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n--- Test {i} ---")
        result = validate_and_generate(test_input)
        print("-" * 40)
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Note: TRACELOOP_API_KEY is also recommended for full functionality
    if not os.getenv("TRACELOOP_API_KEY"):
        print("ℹ️ Note: TRACELOOP_API_KEY not set. Some guardrails features may be limited.")
        print("Set it with: export TRACELOOP_API_KEY='your-traceloop-key'")
    
    main() 