"""
Multi-Agent Routing System

A coordinator-based multi-agent system that intelligently routes tasks between 
local LLM resources and OpenAI API based on task complexity and requirements.
"""

__version__ = "0.1.0"
__author__ = "Multi-Agent Routing System"

from .models import (
    UserRequest,
    TaskClassification,
    RoutingDecision,
    AgentResponse,
    SystemConfig,
    LocalLLMConfig,
    OpenAIConfig,
)

__all__ = [
    "UserRequest",
    "TaskClassification", 
    "RoutingDecision",
    "AgentResponse",
    "SystemConfig",
    "LocalLLMConfig",
    "OpenAIConfig",
]