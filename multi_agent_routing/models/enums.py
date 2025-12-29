"""
Enumerations for the Multi-Agent Routing System.
"""

from enum import Enum, auto


class RequestType(Enum):
    """Types of user requests that can be processed."""
    SIMPLE = auto()
    COMPLEX = auto()
    QUERY = auto()
    CODE_REVIEW = auto()
    CODE_GENERATION = auto()
    WEB_SEARCH = auto()


class RoutingTarget(Enum):
    """Available routing targets for request processing."""
    LOCAL_LLM = auto()
    OPENAI_API = auto()
    CODE_REVIEWER = auto()
    CODE_GENERATOR = auto()


class ResourceStatus(Enum):
    """Status of system resources."""
    AVAILABLE = auto()
    UNAVAILABLE = auto()
    BUSY = auto()
    ERROR = auto()


class APIStatus(Enum):
    """Status of external API services."""
    ACTIVE = auto()
    RATE_LIMITED = auto()
    QUOTA_EXCEEDED = auto()
    ERROR = auto()
    UNAVAILABLE = auto()