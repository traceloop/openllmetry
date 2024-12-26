from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import *

def print_attributes():
    """Print all available GenAI attributes."""
    # Get all constants that start with GEN_AI
    attributes = sorted([attr for attr in globals() if attr.startswith('GEN_AI_')])
    print("Available GenAI Attributes:")
    print("========================")
    for attr in attributes:
        value = globals()[attr]
        print(f"{attr}: {value}")

if __name__ == "__main__":
    print_attributes()
