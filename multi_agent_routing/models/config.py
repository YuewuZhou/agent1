"""
Configuration models for the Multi-Agent Routing System.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import logging


@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    max_concurrent_requests: int = 10
    backoff_base_delay: float = 1.0
    backoff_max_delay: float = 60.0
    max_retries: int = 3


@dataclass
class CostLimitConfig:
    """Configuration for cost management."""
    daily_limit_usd: float = 10.0
    monthly_limit_usd: float = 300.0
    cost_per_token: float = 0.0001
    alert_threshold_percent: float = 80.0


@dataclass
class LocalLLMConfig:
    """Configuration for local LLM resources."""
    model_path: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    max_context_length: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 30
    memory_limit_mb: int = 8192
    max_concurrent_requests: int = 5


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API integration."""
    api_key: str = ""
    model_preferences: Dict[str, str] = field(default_factory=lambda: {
        "simple": "gpt-3.5-turbo",
        "complex": "gpt-4",
        "code": "gpt-4"
    })
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)
    cost_limits: CostLimitConfig = field(default_factory=CostLimitConfig)
    timeout_seconds: int = 60
    max_retries: int = 3


@dataclass
class RoutingThresholds:
    """Thresholds for routing decisions."""
    complexity_threshold: float = 0.7
    confidence_threshold: float = 0.8
    processing_time_threshold: float = 30.0
    fallback_delay_seconds: float = 5.0


@dataclass
class LoggingConfig:
    """Configuration for system logging."""
    level: int = logging.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "multi_agent_routing.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True


@dataclass
class SystemConfig:
    """Main system configuration."""
    local_llm_config: LocalLLMConfig = field(default_factory=LocalLLMConfig)
    openai_config: OpenAIConfig = field(default_factory=OpenAIConfig)
    routing_thresholds: RoutingThresholds = field(default_factory=RoutingThresholds)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    debug_mode: bool = False
    enable_fallback: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)