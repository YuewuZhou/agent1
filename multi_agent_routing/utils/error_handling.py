"""
Error handling utilities and custom exceptions for the Multi-Agent Routing System.
"""

from typing import Optional, Dict, Any


class MultiAgentRoutingError(Exception):
    """Base exception for all Multi-Agent Routing System errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}


class ConfigurationError(MultiAgentRoutingError):
    """Raised when there are configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key


class ResourceUnavailableError(MultiAgentRoutingError):
    """Raised when a required resource is unavailable."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="RESOURCE_UNAVAILABLE", **kwargs)
        self.resource_type = resource_type


class APIError(MultiAgentRoutingError):
    """Raised when external API calls fail."""
    
    def __init__(self, message: str, api_name: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="API_ERROR", **kwargs)
        self.api_name = api_name
        self.status_code = status_code


class ClassificationError(MultiAgentRoutingError):
    """Raised when task classification fails."""
    
    def __init__(self, message: str, request_content: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CLASSIFICATION_ERROR", **kwargs)
        self.request_content = request_content


class FallbackError(MultiAgentRoutingError):
    """Raised when all fallback mechanisms fail."""
    
    def __init__(self, message: str, attempted_targets: Optional[list] = None, **kwargs):
        super().__init__(message, error_code="FALLBACK_ERROR", **kwargs)
        self.attempted_targets = attempted_targets or []


def handle_error(error: Exception, logger=None, context: Optional[Dict[str, Any]] = None) -> MultiAgentRoutingError:
    """
    Convert generic exceptions to MultiAgentRoutingError instances.
    
    Args:
        error: The original exception
        logger: Optional logger for error reporting
        context: Additional context information
        
    Returns:
        MultiAgentRoutingError instance
    """
    if isinstance(error, MultiAgentRoutingError):
        return error
    
    # Convert common exceptions
    if isinstance(error, ValueError):
        routing_error = ConfigurationError(str(error), context=context)
    elif isinstance(error, ConnectionError):
        routing_error = ResourceUnavailableError(str(error), context=context)
    elif isinstance(error, TimeoutError):
        routing_error = ResourceUnavailableError(f"Operation timed out: {str(error)}", context=context)
    else:
        routing_error = MultiAgentRoutingError(str(error), context=context)
    
    if logger:
        logger.log_error(routing_error, context)
    
    return routing_error


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    Decorator for implementing retry logic with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    """
    import time
    import random
    from functools import wraps
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay
                    time.sleep(delay + jitter)
            
            # If we get here, all retries failed
            raise FallbackError(
                f"All retry attempts failed: {str(last_exception)}",
                context={"max_retries": max_retries, "last_error": str(last_exception)}
            )
        
        return wrapper
    
    return decorator