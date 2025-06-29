import re
from typing import Dict, Any, Tuple
from .types import InputSchemaMapping

def extract_input_data(args: Tuple[Any, ...], kwargs: Dict[str, Any], schema: InputSchemaMapping) -> Dict[str, Any]:
    """Extract data from function arguments based on InputSchemaMapping."""
    extracted_data = {}
    
    # Combine args and kwargs into a single data structure
    # For simplicity, we'll treat positional args as indexed keys
    combined_data = {}
    
    # Add positional arguments
    for i, arg in enumerate(args):
        combined_data[f"arg_{i}"] = arg
    
    # Add keyword arguments
    combined_data.update(kwargs)
    
    for field_name, extractor in schema.items():
        value = None
        
        if extractor.source == "input":
            if extractor.key:
                # Direct key lookup
                value = combined_data.get(extractor.key)
            else:
                # Use field name as key
                value = combined_data.get(field_name)
                
        elif extractor.source == "output":
            # For output source, we'll need to handle this differently
            # This would require access to the function's return value
            # For now, we'll skip output extraction
            continue
        
        if value is not None:
            if extractor.use_regex and extractor.regex_pattern:
                # Apply regex pattern
                if isinstance(value, str):
                    match = re.search(extractor.regex_pattern, value)
                    if match:
                        value = match.group(0)
                    else:
                        value = None
            
            if value is not None:
                extracted_data[field_name] = value
    
    return extracted_data

def extract_output_data(result: Any, schema: InputSchemaMapping) -> Dict[str, Any]:
    """Extract data from function result based on InputSchemaMapping."""
    extracted_data = {}
    
    for field_name, extractor in schema.items():
        if extractor.source == "output":
            value = None
            
            if extractor.key:
                # Try to access key from result
                if isinstance(result, dict):
                    value = result.get(extractor.key)
                elif hasattr(result, extractor.key):
                    value = getattr(result, extractor.key)
            else:
                # Use field name as key
                if isinstance(result, dict):
                    value = result.get(field_name)
                elif hasattr(result, field_name):
                    value = getattr(result, field_name)
            
            if value is not None:
                if extractor.use_regex and extractor.regex_pattern:
                    # Apply regex pattern
                    if isinstance(value, str):
                        match = re.search(extractor.regex_pattern, value)
                        if match:
                            value = match.group(0)
                        else:
                            value = None
                
                if value is not None:
                    extracted_data[field_name] = value
    
    return extracted_data 