from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class InputExtractor:
    source: str  # mapping value

InputSchemaMapping = Dict[str, InputExtractor] 