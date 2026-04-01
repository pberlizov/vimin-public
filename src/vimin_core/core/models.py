"""
Model Management Module

Provides model discovery, classification, and management capabilities
for ONNX models used in local NPU execution.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass

from vimin_core.core.task import TaskType, TaskComplexity

@dataclass
class ModelInfo:
    """Information about an available model"""
    name: str
    path: str
    size_mb: float
    task_types: List[str]
    complexity: str
    npu_capable: bool
    description: str
    provider: str = "ONNX"
    performance_score: float = 0.0  # 0-1 score for model performance
    quantized_variant: Optional[str] = None  # Path to INT8/Q4 quantized version

class ModelRegistry:
    """
    Registry for managing ONNX models
    """
    
    def __init__(self, models_dir: str = "models"):
        self.logger = logging.getLogger(__name__)
        self.models_dir = Path(models_dir)
        self.models: Dict[str, ModelInfo] = {}
        self._discover_models()
    
    def _discover_models(self):
        """Discover available models in the models directory"""
        if not self.models_dir.exists():
            self.logger.warning(f"Models directory {self.models_dir} does not exist")
            return
        
        # Look for .onnx files
        for model_file in self.models_dir.glob("**/*.onnx"):
            try:
                # Get file size
                size_mb = model_file.stat().st_size / (1024 * 1024)
                
                # Extract model name from filename
                model_name = model_file.stem
                
                # Determine task types and complexity based on config.json or name
                task_types, complexity = self._classify_model(model_name, model_file)
                
                # Check for an auto-generated hardware profile
                npu_capable = True # default assumption
                profile_path = model_file.parent / "hardware_profile.json"
                if profile_path.exists():
                    try:
                        with open(profile_path, "r") as f:
                            profile = json.load(f)
                        npu_capable = profile.get("npu_capable", True)
                    except Exception:
                        pass
                
                model_info = ModelInfo(
                    name=model_name,
                    path=str(model_file),
                    size_mb=size_mb,
                    task_types=task_types,
                    complexity=complexity,
                    npu_capable=npu_capable,
                    description=f"ONNX model: {model_name} (NPU Capable: {npu_capable})"
                )
                
                self.models[model_name] = model_info
                self.logger.info(f"Discovered model: {model_name} ({size_mb:.1f}MB)")
                
            except Exception as e:
                self.logger.error(f"Error discovering model {model_file}: {e}")
    
    def _classify_model(self, model_name: str, model_file: Path) -> tuple[List[str], str]:
        """
        Classify model based on huggingface config.json if available, else fallback to name
        """
        task_types = []
        complexity = "Medium"
        
        # 1. Try to read from config.json if downloaded by optimum
        config_path = model_file.parent / "config.json"
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    
                arch = config.get("architectures", [""])[0].lower()
                model_type = config.get("model_type", "").lower()
                params = config.get("num_parameters", 0)  # Often missing, but good if there
                
                # Determine Types by strict architecture
                if "causallm" in arch or "gpt" in model_type or "llama" in model_type:
                    task_types.extend(["TEXT_GENERATION", "REASONING"])
                if "sequenceclassification" in arch or "bert" in model_type or "encoder" in arch:
                    task_types.extend(["PII_MASKING", "CLASSIFICATION", "SENTIMENT_ANALYSIS"])
                if "whisper" in model_type or "speech" in model_type:
                    task_types.append("SPEECH_TO_TEXT")
                    
                # Determine complexity by hidden stats
                hidden_size = config.get("hidden_size", 0)
                num_layers = config.get("num_hidden_layers", 0)
                
                if params > 0:
                    if params < 5e8: complexity = "Low"
                    elif params > 5e9: complexity = "High"
                elif hidden_size > 0:
                    if hidden_size < 1024 and num_layers < 12: complexity = "Low"
                    elif hidden_size > 4096: complexity = "High"
                    
            except Exception as e:
                self.logger.warning(f"Error reading config.json for {model_name}: {e}")
                
        # 2. Fallback to name heuristic if config.json not present or not matched
        model_name_lower = model_name.lower()
        if not task_types:
            if any(keyword in model_name_lower for keyword in ["bert", "encoder"]):
                task_types.extend(["PII_MASKING", "CLASSIFICATION", "SENTIMENT_ANALYSIS"])
            if any(keyword in model_name_lower for keyword in ["gpt", "generation", "lm", "llama", "phi", "qwen", "mistral"]):
                task_types.extend(["TEXT_GENERATION", "REASONING"])
            if "summarization" in model_name_lower:
                task_types.append("SUMMARIZATION")
            if "translation" in model_name_lower:
                task_types.append("TRANSLATION")
            if "code" in model_name_lower:
                task_types.append("CODE_GENERATION")
            # Voice model detection
            if any(keyword in model_name_lower for keyword in ["whisper", "stt", "speech", "vosk", "wav2vec"]):
                task_types.append("SPEECH_TO_TEXT")
            if any(keyword in model_name_lower for keyword in ["silero_tts", "tts", "kokoro", "piper", "melotts", "vits"]):
                task_types.append("TEXT_TO_SPEECH")
            if any(keyword in model_name_lower for keyword in ["vad", "silero_vad", "voice_activity"]):
                task_types.append("VOICE_ACTIVITY_DETECTION")
            
            if not task_types:
                task_types = ["TEXT_GENERATION", "REASONING"]
            
            # Name-based complexity
            if any(keyword in model_name_lower for keyword in ["tiny", "small", "mini", "vad", "piper", "1b", "0.5b"]):
                complexity = "Low"
            elif any(keyword in model_name_lower for keyword in ["large", "xlarge", "xl", "7b", "8b", "70b"]):
                complexity = "High"
        
        return task_types, complexity
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """
        Get model information by name
        
        Args:
            name: Model name
            
        Returns:
            ModelInfo or None if not found
        """
        return self.models.get(name)
    
    def list_models(self) -> Dict[str, ModelInfo]:
        """
        Get all available models
        
        Returns:
            Dictionary of all models
        """
        return self.models.copy()
    
    def get_models_by_task_type(self, task_type: str) -> List[ModelInfo]:
        """
        Get models suitable for specific task type
        
        Args:
            task_type: Task type to filter by
            
        Returns:
            List of suitable models
        """
        suitable_models = []
        for model in self.models.values():
            if task_type in model.task_types:
                suitable_models.append(model)
        
        # Sort by performance score (descending)
        suitable_models.sort(key=lambda m: m.performance_score, reverse=True)
        return suitable_models
    
    def get_models_by_complexity(self, complexity: str) -> List[ModelInfo]:
        """
        Get models by complexity level
        
        Args:
            complexity: Complexity level (Low, Medium, High)
            
        Returns:
            List of models with specified complexity
        """
        return [model for model in self.models.values() if model.complexity == complexity]
    
    def get_best_model(self, task_type: str, complexity: str = None) -> Optional[ModelInfo]:
        """
        Get the best model for a task type and optional complexity
        
        Args:
            task_type: Task type
            complexity: Optional complexity level
            
        Returns:
            Best model or None
        """
        candidates = self.get_models_by_task_type(task_type)
        
        if complexity:
            candidates = [m for m in candidates if m.complexity == complexity]
        
        if not candidates:
            return None
        
        # Return model with highest performance score
        return max(candidates, key=lambda m: m.performance_score)
    
    def register_model(self, model_info: ModelInfo):
        """
        Register a new model
        
        Args:
            model_info: Model information
        """
        self.models[model_info.name] = model_info
        self.logger.info(f"Registered model: {model_info.name}")
    
    def update_model_performance(self, name: str, performance_score: float):
        """
        Update model performance score
        
        Args:
            name: Model name
            performance_score: New performance score (0-1)
        """
        if name in self.models:
            self.models[name].performance_score = performance_score
            self.logger.info(f"Updated performance score for {name}: {performance_score}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get statistics about available models
        
        Returns:
            Dictionary with model statistics
        """
        if not self.models:
            return {"message": "No models available"}
        
        total_models = len(self.models)
        total_size_mb = sum(m.size_mb for m in self.models.values())
        
        complexity_counts = {}
        for model in self.models.values():
            complexity_counts[model.complexity] = complexity_counts.get(model.complexity, 0) + 1
        
        task_type_counts = {}
        for model in self.models.values():
            for task_type in model.task_types:
                task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        
        return {
            "total_models": total_models,
            "total_size_mb": total_size_mb,
            "complexity_distribution": complexity_counts,
            "task_type_coverage": task_type_counts,
            "average_model_size_mb": total_size_mb / total_models
        }
