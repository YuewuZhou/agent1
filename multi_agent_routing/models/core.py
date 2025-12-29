"""
Core data models for request processing and routing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from .enums import RequestType, RoutingTarget


@dataclass
class ConversationContext:
    """Context for multi-turn conversations."""
    conversation_id: str
    history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserRequest:
    """Represents a user request to be processed by the system."""
    content: str
    request_type: RequestType
    context: Optional[ConversationContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = "anonymous"


@dataclass
class TaskClassification:
    """Classification result for a user request."""
    complexity_score: float
    requires_web_search: bool
    requires_advanced_reasoning: bool
    estimated_processing_time: float
    recommended_target: RoutingTarget
    confidence_score: float
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Decision made by the central router for request routing."""
    target: RoutingTarget
    reasoning: str
    classification: TaskClassification
    fallback_targets: List[RoutingTarget] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent after processing a request."""
    content: str
    source_agent: str
    processing_time: float
    cost_estimate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)