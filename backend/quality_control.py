"""
Quality control system for validating research outputs and ensuring data quality.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]
    timestamp: str

class QualityController:
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "quality_control"
        self.validators = []
        self.quality_metrics = {}
        self._ensure_storage_path()

    def _ensure_storage_path(self):
        """Ensure the storage path exists."""
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)

    def add_validator(self, validator_func):
        """Add a new validator function."""
        self.validators.append(validator_func)

    async def validate(self, data: Any) -> ValidationResult:
        """Run all validators on the data."""
        issues = []
        suggestions = []
        metadata = {}

        for validator in self.validators:
            try:
                result = await validator(data)
                if not result['is_valid']:
                    issues.extend(result.get('issues', []))
                    suggestions.extend(result.get('suggestions', []))
                metadata.update(result.get('metadata', {}))
            except Exception as e:
                logger.error(f"Error in validator: {str(e)}")
                issues.append(f"Validator error: {str(e)}")

        validation_result = ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            suggestions=suggestions,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )

        self._save_validation_result(validation_result)
        return validation_result

    def _save_validation_result(self, result: ValidationResult):
        """Save validation result to storage."""
        try:
            result_file = Path(self.storage_path) / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'is_valid': result.is_valid,
                    'issues': result.issues,
                    'suggestions': result.suggestions,
                    'metadata': result.metadata,
                    'timestamp': result.timestamp
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving validation result: {str(e)}")

    def get_validation_history(self) -> List[ValidationResult]:
        """Get validation history from storage."""
        results = []
        try:
            for file in Path(self.storage_path).glob("validation_*.json"):
                with open(file, 'r') as f:
                    data = json.load(f)
                    results.append(ValidationResult(**data))
        except Exception as e:
            logger.error(f"Error loading validation history: {str(e)}")
        return results

    def clear_validation_history(self):
        """Clear validation history."""
        try:
            for file in Path(self.storage_path).glob("validation_*.json"):
                file.unlink()
        except Exception as e:
            logger.error(f"Error clearing validation history: {str(e)}")

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get current quality metrics."""
        return self.quality_metrics

    def update_quality_metrics(self, metrics: Dict[str, Any]):
        """Update quality metrics."""
        self.quality_metrics.update(metrics)
        self._save_quality_metrics()

    def _save_quality_metrics(self):
        """Save quality metrics to storage."""
        try:
            metrics_file = Path(self.storage_path) / "quality_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.quality_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving quality metrics: {str(e)}")

    def load_quality_metrics(self):
        """Load quality metrics from storage."""
        try:
            metrics_file = Path(self.storage_path) / "quality_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.quality_metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading quality metrics: {str(e)}")

    async def validate_research_output(self, output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate research output specifically."""
        validation_result = await self.validate(output)
        
        # Update quality metrics
        self.update_quality_metrics({
            'total_validations': self.quality_metrics.get('total_validations', 0) + 1,
            'successful_validations': self.quality_metrics.get('successful_validations', 0) + (1 if validation_result.is_valid else 0),
            'last_validation': validation_result.timestamp
        })

        return validation_result.is_valid, validation_result.issues 