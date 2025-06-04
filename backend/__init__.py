"""
Deep Researcher backend package.
"""

from .config import (
    OPENAI_API_KEY,
    VECTORSTORE_CONFIGS,
    VECTORSTORE_DIR,
    BASE_DIR,
    EXPORTS_DIR,
    LOGS_DIR
)

__all__ = [
    'OPENAI_API_KEY',
    'VECTORSTORE_CONFIGS',
    'VECTORSTORE_DIR',
    'BASE_DIR',
    'EXPORTS_DIR',
    'LOGS_DIR'
] 