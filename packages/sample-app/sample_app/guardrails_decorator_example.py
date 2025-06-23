"""
Guardrails Decorator Example

This example demonstrates how to easily add guardrails to existing functions
using decorators, making it simple to integrate safety checks into your code.
"""

import os
from openai import OpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.guardrails import GuardrailsDecorator
from traceloop.sdk.decorators import task, workflow

# Initialize OpenAI and Traceloop
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
Traceloop.init(app_name="guardrails_decorator_demo")

# Get guardrails decorator
guardrails = GuardrailsDecorator(Traceloop.get().guardrails)


@task(name="basic_text_generation")
def generate_basic_text(prompt: str) -> str:
    """Basic text generation without guardrails"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content or ""


@guardrails.validate_input("content-safety")
@task(name="safe_text_generation")
def generate_safe_text(prompt: str) -> str:
    """Text generation with input validation"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content or ""


@guardrails.validate_output("output-quality")
@task(name="quality_checked_generation")
def generate_quality_checked_text(prompt: str) -> str:
    """Text generation with output quality validation"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response.choices[0].message.content or ""


@guardrails.validate_input("input-safety")
@guardrails.validate_output("output-safety")
@task(name="fully_protected_generation")
def generate_fully_protected_text(prompt: str) -> str:
    """Text generation with both input and output validation"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message.content or ""


@task(name="story_generation")
def generate_story(topic: str, style: str = "adventure") -> str:
    """Generate a story with specific topic and style"""
    prompt = f"Write a short {style} story about {topic}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.8
    )
    return response.choices[0].message.content or ""


@guardrails.validate_input("story-appropriateness")
@guardrails.validate_output("story-quality")
@task(name="safe_story_generation")
def generate_safe_story(topic: str, style: str = "adventure") -> str:
    """Generate a story with guardrails protection"""
    prompt = f"Write a short {style} story about {topic}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.8
    )
    return response.choices[0].message.content or ""


def demo_basic_vs_protected():
    """Demonstrate the difference between basic and protected functions"""
    print("🔄 Comparing Basic vs Protected Text Generation")
    print("=" * 50)
    
    test_prompt = "Tell me about space exploration"
    
    print("\n--- Basic Generation (No Guardrails) ---")
    try:
        basic_result = generate_basic_text(test_prompt)
        print(f"✅ Generated: {basic_result[:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n--- Protected Generation (With Input Validation) ---")
    try:
        safe_result = generate_safe_text(test_prompt)
        print(f"✅ Generated: {safe_result[:100]}...")
    except Exception as e:
        print(f"❌ Blocked or Error: {e}")
    
    print("\n--- Fully Protected (Input + Output Validation) ---")
    try:
        protected_result = generate_fully_protected_text(test_prompt)
        print(f"✅ Generated: {protected_result[:100]}...")
    except Exception as e:
        print(f"❌ Blocked or Error: {e}")


def demo_story_generation():
    """Demonstrate story generation with different safety levels"""
    print("\n🎭 Story Generation Demo")
    print("=" * 30)
    
    test_cases = [
        ("robots", "sci-fi"),
        ("friendship", "heartwarming"),
        ("mystery", "detective"),
        ("adventure", "fantasy")
    ]
    
    for topic, style in test_cases:
        print(f"\n📖 Story: {style} about {topic}")
        
        # Basic story generation
        print("--- Without Guardrails ---")
        try:
            story = generate_story(topic, style)
            print(f"Story length: {len(story)} characters")
            print(f"Preview: {story[:80]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Protected story generation
        print("--- With Guardrails ---")
        try:
            safe_story = generate_safe_story(topic, style)
            print(f"✅ Safe story length: {len(safe_story)} characters")
            print(f"Preview: {safe_story[:80]}...")
        except Exception as e:
            print(f"❌ Blocked or Error: {e}")


def demo_error_handling():
    """Demonstrate how guardrails handle problematic content"""
    print("\n⚠️ Error Handling Demo")
    print("=" * 25)
    
    # Test with potentially problematic inputs
    problematic_inputs = [
        "Write about cooking pasta",  # Safe content
        "Explain quantum computing",  # Safe content
        "Tell me about historical events",  # Safe content
    ]
    
    for test_input in problematic_inputs:
        print(f"\n🧪 Testing: '{test_input}'")
        
        try:
            result = generate_fully_protected_text(test_input)
            print(f"✅ Passed: {result[:60]}...")
        except Exception as e:
            print(f"❌ Handled: {str(e)[:100]}...")


@workflow(name="guardrails_decorator_workflow")
def main_workflow():
    """Main workflow demonstrating decorator-based guardrails"""
    print("🚀 Guardrails Decorator Examples")
    print("=" * 40)
    
    # Set tracking properties
    Traceloop.set_association_properties({
        "demo_type": "decorator_examples",
        "user_id": "demo_user"
    })
    
    # Run demonstrations
    demo_basic_vs_protected()
    demo_story_generation()
    demo_error_handling()
    
    print("\n🎉 All decorator examples completed!")


def main():
    """Main function"""
    print("🛡️ Starting Guardrails Decorator Demo")
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required!")
        return
    
    if not os.getenv("TRACELOOP_API_KEY"):
        print("ℹ️ TRACELOOP_API_KEY not set - some features may be limited")
    
    # Run the main workflow
    main_workflow()


if __name__ == "__main__":
    main() 