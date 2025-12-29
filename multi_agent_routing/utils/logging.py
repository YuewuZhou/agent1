"""
Logging utilities for the Multi-Agent Routing System.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from ..models.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """
    Set up logging configuration for the system.
    
    Args:
        config: Logging configuration settings
    """
    # Create root logger
    root_logger = logging.getLogger("multi_agent_routing")
    root_logger.setLevel(config.level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.format)
    
    # Console handler
    if config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if config.enable_file and config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config.file_path,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(config.level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"multi_agent_routing.{name}")


class RoutingLogger:
    """
    Specialized logger for routing decisions and system events.
    """
    
    def __init__(self, name: str = "routing"):
        self.logger = get_logger(name)
    
    def log_routing_decision(self, decision_data: dict) -> None:
        """Log a routing decision with structured data."""
        self.logger.info(
            "Routing decision made",
            extra={
                "event_type": "routing_decision",
                "data": decision_data
            }
        )
    
    def log_fallback(self, original_target: str, fallback_target: str, reason: str) -> None:
        """Log a fallback event."""
        self.logger.warning(
            f"Fallback triggered: {original_target} -> {fallback_target}",
            extra={
                "event_type": "fallback",
                "original_target": original_target,
                "fallback_target": fallback_target,
                "reason": reason
            }
        )
    
    def log_error(self, error: Exception, context: Optional[dict] = None) -> None:
        """Log an error with optional context."""
        self.logger.error(
            f"Error occurred: {str(error)}",
            extra={
                "event_type": "error",
                "error_type": type(error).__name__,
                "context": context or {}
            },
            exc_info=True
        )