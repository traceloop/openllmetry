from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class InputExtractor:
    source: str  # "input" or "output"
    key: Optional[str] = None  # Key to extract from
    use_regex: bool = False  # Whether to use regex pattern
    regex_pattern: Optional[str] = None  # Regex pattern to apply

InputSchemaMapping = Dict[str, InputExtractor] 