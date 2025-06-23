import os
import asyncio
from openai import OpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.guardrails import (
    GuardrailsClient,
    GuardrailsDecorator,
    GuardrailInputData,
    GuardrailConfig,
    GuardrailAction,
    GuardrailResult,
)
from traceloop.sdk.decorators import task, workflow

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Traceloop
Traceloop.init(app_name="guardrails_demo")

# Get the guardrails client
guardrails_client = Traceloop.get().guardrails

# Initialize guardrails decorator
guardrails_decorator = GuardrailsDecorator(guardrails_client)


@task(name="text_generation")
def generate_text(prompt: str) -> str:
    """Generate text using OpenAI API"""
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    return completion.choices[0].message.content or ""


@guardrails_decorator.validate_input("toxicity-checker")
@task(name="safe_text_generation")
def safe_generate_text(prompt: str) -> str:
    """Generate text with input validation using guardrails decorator"""
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    return completion.choices[0].message.content or ""


@guardrails_decorator.validate_output("content-safety-checker")
@task(name="content_filtered_generation")
def content_filtered_generate_text(prompt: str) -> str:
    """Generate text with output validation using guardrails decorator"""
    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    return completion.choices[0].message.content or ""


async def manual_input_validation_example():
    """Example of manual input validation using guardrails client"""
    print("\n=== Manual Input Validation Example ===")
    
    # Example user input that might be problematic
    user_input = "Write a story about violence"
    
    # Create input data for guardrails
    input_data = GuardrailInputData(
        content=user_input,
        context={"user_id": "user_123", "session_id": "session_456"},
        metadata={"source": "user_input", "timestamp": "2024-01-01T10:00:00Z"}
    )
    
    # Configure guardrails
    config = GuardrailConfig(
        thresholds={"toxicity": 0.7, "violence": 0.8},
        parameters={"check_categories": ["toxicity", "hate", "violence"]},
        settings={"strict_mode": True}
    )
    
    try:
        # Validate input using guardrails
        result = await guardrails_client.validate_input(
            evaluator_slug="content-safety-checker",
            input_data=input_data,
            config=config,
            timeout=30
        )
        
        print(f"Input validation result: {result.action}")
        print(f"Score: {result.score}")
        print(f"Reason: {result.reason}")
        
        if result.pass_through:
            print("✅ Input is safe, proceeding with text generation...")
            generated_text = generate_text(user_input)
            print(f"Generated text: {generated_text}")
        elif result.blocked:
            print("❌ Input blocked by guardrails")
            print(f"Reason: {result.reason}")
        elif result.retry_required:
            print("⚠️ Input needs retry or modification")
            print(f"Reason: {result.reason}")
            
    except Exception as e:
        print(f"Error during guardrails validation: {e}")


async def manual_output_validation_example():
    """Example of manual output validation using guardrails client"""
    print("\n=== Manual Output Validation Example ===")
    
    # Generate some content first
    prompt = "Write a short story about a robot"
    generated_content = generate_text(prompt)
    
    print(f"Generated content: {generated_content}")
    
    # Create output data for validation
    output_data = GuardrailInputData(
        content=generated_content,
        context={"original_prompt": prompt, "model": "gpt-3.5-turbo"},
        metadata={"generation_time": "2024-01-01T10:00:00Z"}
    )
    
    # Configure output validation
    config = GuardrailConfig(
        thresholds={"appropriateness": 0.8, "relevance": 0.7},
        parameters={"check_relevance": True, "check_appropriateness": True},
        settings={"detailed_analysis": True}
    )
    
    try:
        # Validate output using guardrails
        result = await guardrails_client.validate_output(
            evaluator_slug="output-quality-checker",
            output_data=output_data,
            config=config,
            timeout=30
        )
        
        print(f"Output validation result: {result.action}")
        print(f"Score: {result.score}")
        print(f"Reason: {result.reason}")
        
        if result.pass_through:
            print("✅ Output is appropriate and safe")
        elif result.blocked:
            print("❌ Output blocked by guardrails")
            print("Need to regenerate or modify content")
        elif result.retry_required:
            print("⚠️ Output needs improvement")
            print("Consider regenerating with different parameters")
            
    except Exception as e:
        print(f"Error during output validation: {e}")


def decorator_input_validation_example():
    """Example of using guardrails decorator for input validation"""
    print("\n=== Decorator Input Validation Example ===")
    
    test_inputs = [
        "Tell me a joke about programming",
        "Write a positive review about a restaurant",
        "Explain quantum physics in simple terms"
    ]
    
    for test_input in test_inputs:
        try:
            print(f"\nTesting input: '{test_input}'")
            result = safe_generate_text(test_input)
            print(f"✅ Input validated successfully")
            print(f"Generated text: {result}")
        except Exception as e:
            print(f"❌ Input validation failed: {e}")


def decorator_output_validation_example():
    """Example of using guardrails decorator for output validation"""
    print("\n=== Decorator Output Validation Example ===")
    
    test_prompts = [
        "Write a haiku about nature",
        "Explain the benefits of exercise",
        "Describe a beautiful sunset"
    ]
    
    for prompt in test_prompts:
        try:
            print(f"\nTesting prompt: '{prompt}'")
            result = content_filtered_generate_text(prompt)
            print(f"✅ Output validated successfully")
            print(f"Generated text: {result}")
        except Exception as e:
            print(f"❌ Output validation failed: {e}")


@workflow(name="guardrails_workflow")
async def comprehensive_guardrails_workflow():
    """A comprehensive workflow demonstrating different guardrails use cases"""
    print("🛡️ Starting Comprehensive Guardrails Workflow")
    
    # Set association properties for tracking
    Traceloop.set_association_properties({
        "user_id": "demo_user_123",
        "session_id": "demo_session_456",
        "workflow": "guardrails_demo"
    })
    
    # Example 1: Manual input validation
    await manual_input_validation_example()
    
    # Example 2: Manual output validation
    await manual_output_validation_example()
    
    # Example 3: Decorator-based input validation
    decorator_input_validation_example()
    
    # Example 4: Decorator-based output validation
    decorator_output_validation_example()
    
    print("\n🎉 Guardrails workflow completed!")


def sync_guardrails_example():
    """Example of using synchronous guardrails validation"""
    print("\n=== Synchronous Guardrails Example ===")
    
    # Test input
    test_input = "Write a tutorial on web development"
    
    # Create input data
    input_data = GuardrailInputData(
        content=test_input,
        context={"type": "tutorial_request"},
        metadata={"sync_example": True}
    )
    
    # Synchronous validation
    try:
        result = guardrails_client.validate_input_sync(
            evaluator_slug="content-appropriateness",
            input_data=input_data,
            timeout=15
        )
        
        print(f"Sync validation result: {result.action}")
        print(f"Score: {result.score}")
        
        if result.pass_through:
            print("✅ Sync validation passed")
            # Proceed with text generation
            generated_text = generate_text(test_input)
            print(f"Generated: {generated_text[:100]}...")
        else:
            print(f"❌ Sync validation failed: {result.reason}")
            
    except Exception as e:
        print(f"Error in sync validation: {e}")


def custom_guardrail_progress_callback():
    """Example with custom progress callback"""
    print("\n=== Custom Progress Callback Example ===")
    
    def on_progress(event_data):
        """Custom progress callback function"""
        print(f"📊 Progress: {event_data.get('status', 'processing')}...")
    
    # Test with progress callback
    test_input = "Explain artificial intelligence"
    input_data = GuardrailInputData(content=test_input)
    
    try:
        result = guardrails_client.validate_input_sync(
            evaluator_slug="ai-content-validator",
            input_data=input_data,
            on_progress=on_progress,
            timeout=20
        )
        
        print(f"Validation with progress completed: {result.action}")
        
    except Exception as e:
        print(f"Error with progress callback: {e}")


async def main():
    """Main function to run all guardrails examples"""
    print("🚀 Starting Guardrails Sample Application")
    print("=" * 50)
    
    # Run the comprehensive workflow
    await comprehensive_guardrails_workflow()
    
    # Run synchronous examples
    sync_guardrails_example()
    
    # Run custom callback example
    custom_guardrail_progress_callback()
    
    print("\n✨ All guardrails examples completed!")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
        print("You can also set TRACELOOP_API_KEY for full guardrails functionality")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    # Run the main async function
    asyncio.run(main()) 