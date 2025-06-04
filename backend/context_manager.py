"""
Enhanced context management system for maintaining state and history across the research pipeline.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ContextState:
    query: str
    timestamp: str
    state: Dict[str, Any]
    metadata: Dict[str, Any]

class ContextManager:
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "context_storage"
        self.current_state: Optional[ContextState] = None
        self.history: List[ContextState] = []
        self._ensure_storage_path()

    def _ensure_storage_path(self):
        """Ensure the storage path exists."""
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

    def update_context(self, query: str, state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Update the current context with new state."""
        self.current_state = ContextState(
            query=query,
            timestamp=datetime.now().isoformat(),
            state=state,
            metadata=metadata or {}
        )
        self.history.append(self.current_state)
        self._save_state()

    def get_current_state(self) -> Optional[ContextState]:
        """Get the current context state."""
        return self.current_state

    def get_history(self) -> List[ContextState]:
        """Get the complete context history."""
        return self.history

    def clear_context(self):
        """Clear the current context and history."""
        self.current_state = None
        self.history = []
        self._clear_storage()

    def _save_state(self):
        """Save the current state to storage."""
        if not self.current_state:
            return

        try:
            state_file = Path(self.storage_path) / f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(state_file, 'w') as f:
                json.dump(asdict(self.current_state), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving context state: {str(e)}")

    def _clear_storage(self):
        """Clear all stored states."""
        try:
            for file in Path(self.storage_path).glob("state_*.json"):
                file.unlink()
        except Exception as e:
            logger.error(f"Error clearing context storage: {str(e)}")

    def load_state(self, state_id: str) -> Optional[ContextState]:
        """Load a specific state from storage."""
        try:
            state_file = Path(self.storage_path) / f"state_{state_id}.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                return ContextState(**state_data)
        except Exception as e:
            logger.error(f"Error loading context state: {str(e)}")
        return None

    async def async_update_context(self, query: str, state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Asynchronous version of update_context."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.update_context, query, state, metadata)

    def get_state_by_timestamp(self, timestamp: str) -> Optional[ContextState]:
        """Get a state from history by timestamp."""
        for state in self.history:
            if state.timestamp == timestamp:
                return state
        return None

    def get_states_by_query(self, query: str) -> List[ContextState]:
        """Get all states from history for a specific query."""
        return [state for state in self.history if state.query == query]

    def merge_states(self, states: List[ContextState]) -> Dict[str, Any]:
        """Merge multiple states into a single state dictionary."""
        merged_state = {}
        for state in states:
            merged_state.update(state.state)
        return merged_state 