"""
Configuration management for the Multi-Agent Routing System.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from ..models.config import SystemConfig, LocalLLMConfig, OpenAIConfig, LoggingConfig
from .error_handling import ConfigurationError


class ConfigManager:
    """
    Manages system configuration loading, validation, and updates.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self._config: Optional[SystemConfig] = None
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> SystemConfig:
        """
        Load configuration from file or create default configuration.
        
        Returns:
            SystemConfig instance
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                self._config = self._dict_to_config(config_data)
                self.logger.info(f"Configuration loaded from {self.config_path}")
            else:
                self._config = SystemConfig()
                self.save_config()  # Save default config
                self.logger.info("Default configuration created")
            
            self._validate_config(self._config)
            return self._config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def save_config(self, config: Optional[SystemConfig] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save (uses current config if None)
            
        Raises:
            ConfigurationError: If configuration saving fails
        """
        try:
            config_to_save = config or self._config
            if not config_to_save:
                raise ConfigurationError("No configuration to save")
            
            config_dict = self._config_to_dict(config_to_save)
            
            # Ensure directory exists
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def get_config(self) -> SystemConfig:
        """
        Get current configuration, loading if necessary.
        
        Returns:
            SystemConfig instance
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> SystemConfig:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            Updated SystemConfig instance
        """
        current_config = self.get_config()
        config_dict = self._config_to_dict(current_config)
        
        # Apply updates
        self._deep_update(config_dict, updates)
        
        # Convert back to config object
        updated_config = self._dict_to_config(config_dict)
        self._validate_config(updated_config)
        
        self._config = updated_config
        self.save_config()
        
        return updated_config
    
    def _validate_config(self, config: SystemConfig) -> None:
        """
        Validate configuration settings.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Validate OpenAI API key if provided
        if config.openai_config.api_key and not config.openai_config.api_key.startswith('sk-'):
            self.logger.warning("OpenAI API key format may be invalid")
        
        # Validate local LLM configuration
        if config.local_llm_config.timeout_seconds <= 0:
            raise ConfigurationError("Local LLM timeout must be positive")
        
        if config.local_llm_config.max_context_length <= 0:
            raise ConfigurationError("Local LLM max context length must be positive")
        
        # Validate routing thresholds
        if not 0 <= config.routing_thresholds.complexity_threshold <= 1:
            raise ConfigurationError("Complexity threshold must be between 0 and 1")
        
        if not 0 <= config.routing_thresholds.confidence_threshold <= 1:
            raise ConfigurationError("Confidence threshold must be between 0 and 1")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig object."""
        # This is a simplified conversion - in practice, you might want to use
        # a more robust serialization library like pydantic or dataclasses-json
        return SystemConfig(
            local_llm_config=LocalLLMConfig(**config_dict.get('local_llm_config', {})),
            openai_config=OpenAIConfig(**config_dict.get('openai_config', {})),
            logging_config=LoggingConfig(**config_dict.get('logging_config', {})),
            debug_mode=config_dict.get('debug_mode', False),
            enable_fallback=config_dict.get('enable_fallback', True),
            metadata=config_dict.get('metadata', {})
        )
    
    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Convert SystemConfig object to dictionary."""
        return {
            'local_llm_config': config.local_llm_config.__dict__,
            'openai_config': {
                **config.openai_config.__dict__,
                'rate_limits': config.openai_config.rate_limits.__dict__,
                'cost_limits': config.openai_config.cost_limits.__dict__,
            },
            'routing_thresholds': config.routing_thresholds.__dict__,
            'logging_config': config.logging_config.__dict__,
            'debug_mode': config.debug_mode,
            'enable_fallback': config.enable_fallback,
            'metadata': config.metadata
        }
    
    def _deep_update(self, base_dict: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Recursively update nested dictionary."""
        for key, value in updates.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value