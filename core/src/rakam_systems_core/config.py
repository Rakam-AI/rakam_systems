from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass
class Settings:
    env: str = os.environ.get("AI_ENV", "dev")
    debug: bool = os.environ.get("AI_DEBUG", "0") == "1"
    # Add more keys here as needed, always optional to avoid deps.
    # Example: OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

settings = Settings()
