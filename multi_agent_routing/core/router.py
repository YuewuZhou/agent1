"""
Central Router implementation for intelligent request routing.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from ..models import UserRequest, AgentResponse, RoutingDecision, TaskClassification
from ..models.enums import RoutingTarget, RequestType
from ..utils import get_logger
from ..utils.error_handling import (
    ResourceUnavailableError, FallbackError, handle_error, retry_with_backoff
)
from .classifier import TaskClassifier
from .interfaces import LocalLLMInterface, OpenAIInterface
from .agents import CodeReviewer, CodeGenerator


class CentralRouter:
    """
    Central router for intelligent request routing.
    
    Analyzes incoming requests and routes them to the most appropriate resource
    based on task complexity, requirements, and resource availability.
    Implements comprehensive fallback mechanisms and maintains routing audit trail.
    """
    
    def __init__(self, local_llm: Optional[LocalLLMInterface] = None,
                 openai_api: Optional[OpenAIInterface] = None,
                 code_reviewer: Optional[CodeReviewer] = None,
                 code_generator: Optional[CodeGenerator] = None):
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.classifier = TaskClassifier()
        self.local_llm = local_llm or LocalLLMInterface()
        self.openai_api = openai_api or OpenAIInterface()
        self.code_reviewer = code_reviewer or CodeReviewer()
        self.code_generator = code_generator or CodeGenerator()
        
        # Routing decision log for audit trail
        self.routing_log: List[RoutingDecision] = []
        self._max_log_entries = 1000  # Limit log size
        
        # Performance tracking
        self._routing_stats = {
            'total_requests': 0,
            'successful_routes': 0,
            'fallback_routes': 0,
            'failed_routes': 0,
            'target_usage': {target.name: 0 for target in RoutingTarget}
        }
        
        self.logger.info("CentralRouter initialized with all components")
    
    def route_request(self, request: UserRequest) -> AgentResponse:
        """
        Route a user request to the appropriate agent.
        
        Args:
            request: The user request to route and process
            
        Returns:
            AgentResponse: Response from the selected agent
        """
        start_time = time.time()
        self._routing_stats['total_requests'] += 1
        
        try:
            self.logger.info(f"Routing request: {request.content[:50]}...")
            
            # Step 1: Analyze and classify the request
            classification = self.classifier.classify_task(request)
            
            # Step 2: Make routing decision
            routing_decision = self.get_routing_decision(classification)
            
            # Step 3: Log the routing decision
            self.log_routing_decision(routing_decision)
            
            # Step 4: Execute the routing decision
            response = self._execute_routing_decision(routing_decision, request)
            
            # Step 5: Handle fallback if primary routing failed
            if not response.success:
                self.logger.warning(f"Primary routing to {routing_decision.target.name} failed, attempting fallback")
                response = self.handle_fallback(routing_decision.target, request)
            
            # Update statistics
            if response.success:
                self._routing_stats['successful_routes'] += 1
            else:
                self._routing_stats['failed_routes'] += 1
            
            # Add routing metadata to response
            response.metadata.update({
                'routing_decision': {
                    'target': routing_decision.target.name,
                    'reasoning': routing_decision.reasoning,
                    'classification': {
                        'complexity_score': classification.complexity_score,
                        'requires_web_search': classification.requires_web_search,
                        'requires_advanced_reasoning': classification.requires_advanced_reasoning,
                        'confidence_score': classification.confidence_score
                    }
                },
                'total_processing_time': time.time() - start_time
            })
            
            self.logger.info(f"Request routed successfully to {routing_decision.target.name} in {time.time() - start_time:.2f}s")
            return response
            
        except Exception as e:
            self._routing_stats['failed_routes'] += 1
            error = handle_error(e, self.logger, {'request_content': request.content[:100]})
            
            self.logger.error(f"Routing failed: {str(error)}")
            return AgentResponse(
                content=f"Routing failed: {str(error)}",
                source_agent="CentralRouter",
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(error)
            )
    
    def get_routing_decision(self, classification: TaskClassification) -> RoutingDecision:
        """
        Make routing decision based on task classification.
        
        Args:
            classification: Task classification results
            
        Returns:
            RoutingDecision: Decision on where to route the request
        """
        try:
            # Primary target from classification
            primary_target = classification.recommended_target
            
            # Check resource availability and adjust if needed
            available_target = self._check_and_adjust_target(primary_target)
            
            # Determine fallback targets
            fallback_targets = self._determine_fallback_targets(available_target, classification)
            
            # Generate reasoning for the decision
            reasoning = self._generate_routing_reasoning(
                classification, primary_target, available_target, fallback_targets
            )
            
            decision = RoutingDecision(
                target=available_target,
                reasoning=reasoning,
                classification=classification,
                fallback_targets=fallback_targets,
                metadata={
                    'original_recommendation': primary_target.name,
                    'adjusted_target': available_target.name,
                    'availability_check_performed': True
                }
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error making routing decision: {str(e)}")
            # Fallback to safe default
            return RoutingDecision(
                target=RoutingTarget.LOCAL_LLM,
                reasoning=f"Default routing due to decision error: {str(e)}",
                classification=classification,
                fallback_targets=[RoutingTarget.OPENAI_API]
            )
    
    def handle_fallback(self, failed_target: RoutingTarget, request: UserRequest) -> AgentResponse:
        """
        Handle fallback when primary routing target fails.
        
        Args:
            failed_target: The target that failed
            request: The original request
            
        Returns:
            AgentResponse: Response from fallback target or error response
        """
        self._routing_stats['fallback_routes'] += 1
        
        try:
            # Determine fallback sequence based on failed target
            fallback_sequence = self._get_fallback_sequence(failed_target, request)
            
            self.logger.info(f"Attempting fallback sequence: {[t.name for t in fallback_sequence]}")
            
            for fallback_target in fallback_sequence:
                try:
                    self.logger.info(f"Trying fallback to {fallback_target.name}")
                    
                    # Check if fallback target is available
                    if not self._is_target_available(fallback_target):
                        self.logger.warning(f"Fallback target {fallback_target.name} not available")
                        continue
                    
                    # Execute fallback routing
                    response = self._route_to_target(fallback_target, request)
                    
                    if response.success:
                        self.logger.info(f"Fallback to {fallback_target.name} successful")
                        response.metadata['fallback_used'] = True
                        response.metadata['original_target'] = failed_target.name
                        response.metadata['fallback_target'] = fallback_target.name
                        return response
                    else:
                        self.logger.warning(f"Fallback to {fallback_target.name} failed: {response.error_message}")
                        
                except Exception as e:
                    self.logger.error(f"Fallback to {fallback_target.name} error: {str(e)}")
                    continue
            
            # All fallbacks failed
            raise FallbackError(
                "All fallback mechanisms failed",
                attempted_targets=[failed_target] + fallback_sequence
            )
            
        except Exception as e:
            error = handle_error(e, self.logger)
            return AgentResponse(
                content=f"All routing attempts failed: {str(error)}",
                source_agent="CentralRouter",
                processing_time=0.0,
                success=False,
                error_message=str(error),
                metadata={
                    'failed_target': failed_target.name,
                    'fallback_attempted': True,
                    'total_failure': True
                }
            )
    
    def log_routing_decision(self, decision: RoutingDecision) -> None:
        """
        Log routing decision for transparency and debugging.
        
        Args:
            decision: The routing decision to log
        """
        try:
            # Add to routing log
            self.routing_log.append(decision)
            
            # Maintain log size limit
            if len(self.routing_log) > self._max_log_entries:
                self.routing_log = self.routing_log[-self._max_log_entries//2:]  # Keep last half
            
            # Update target usage statistics
            self._routing_stats['target_usage'][decision.target.name] += 1
            
            # Log decision details
            self.logger.info(
                f"Routing decision logged: {decision.target.name} "
                f"(confidence: {decision.classification.confidence_score:.2f}, "
                f"complexity: {decision.classification.complexity_score:.2f})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log routing decision: {str(e)}")
    
    def _execute_routing_decision(self, decision: RoutingDecision, request: UserRequest) -> AgentResponse:
        """Execute the routing decision by calling the appropriate target."""
        try:
            return self._route_to_target(decision.target, request)
        except Exception as e:
            error = handle_error(e, self.logger)
            return AgentResponse(
                content=f"Routing execution failed: {str(error)}",
                source_agent="CentralRouter",
                processing_time=0.0,
                success=False,
                error_message=str(error)
            )
    
    def _route_to_target(self, target: RoutingTarget, request: UserRequest) -> AgentResponse:
        """Route request to specific target."""
        if target == RoutingTarget.LOCAL_LLM:
            return self.local_llm.generate_response(request)
        elif target == RoutingTarget.OPENAI_API:
            return self.openai_api.generate_response(request)
        elif target == RoutingTarget.CODE_REVIEWER:
            return self.code_reviewer.review_code(request)
        elif target == RoutingTarget.CODE_GENERATOR:
            return self.code_generator.generate_code(request)
        else:
            raise ValueError(f"Unknown routing target: {target}")
    
    def _check_and_adjust_target(self, target: RoutingTarget) -> RoutingTarget:
        """Check target availability and adjust if necessary."""
        if self._is_target_available(target):
            return target
        
        # If target not available, find best alternative
        if target == RoutingTarget.LOCAL_LLM:
            return RoutingTarget.OPENAI_API if self._is_target_available(RoutingTarget.OPENAI_API) else target
        elif target == RoutingTarget.OPENAI_API:
            return RoutingTarget.LOCAL_LLM if self._is_target_available(RoutingTarget.LOCAL_LLM) else target
        
        # For specialized agents, fallback to general purpose
        return RoutingTarget.OPENAI_API if self._is_target_available(RoutingTarget.OPENAI_API) else RoutingTarget.LOCAL_LLM
    
    def _is_target_available(self, target: RoutingTarget) -> bool:
        """Check if a routing target is available."""
        try:
            if target == RoutingTarget.LOCAL_LLM:
                return self.local_llm.is_healthy()
            elif target == RoutingTarget.OPENAI_API:
                return self.openai_api.is_healthy()
            elif target in [RoutingTarget.CODE_REVIEWER, RoutingTarget.CODE_GENERATOR]:
                # Specialized agents are always available (they're local)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking availability for {target.name}: {str(e)}")
            return False
    
    def _determine_fallback_targets(self, primary_target: RoutingTarget, 
                                  classification: TaskClassification) -> List[RoutingTarget]:
        """Determine fallback targets based on primary target and classification."""
        fallbacks = []
        
        if primary_target == RoutingTarget.LOCAL_LLM:
            # Local LLM fallback: try OpenAI API
            fallbacks.append(RoutingTarget.OPENAI_API)
        elif primary_target == RoutingTarget.OPENAI_API:
            # OpenAI API fallback: try Local LLM if task is compatible
            if not classification.requires_web_search and classification.complexity_score < 0.8:
                fallbacks.append(RoutingTarget.LOCAL_LLM)
        elif primary_target in [RoutingTarget.CODE_REVIEWER, RoutingTarget.CODE_GENERATOR]:
            # Specialized agent fallback: try general purpose agents
            fallbacks.extend([RoutingTarget.OPENAI_API, RoutingTarget.LOCAL_LLM])
        
        return fallbacks
    
    def _get_fallback_sequence(self, failed_target: RoutingTarget, request: UserRequest) -> List[RoutingTarget]:
        """Get fallback sequence for a failed target."""
        if failed_target == RoutingTarget.LOCAL_LLM:
            return [RoutingTarget.OPENAI_API]
        elif failed_target == RoutingTarget.OPENAI_API:
            # Only fallback to Local LLM if request doesn't require web search
            if not self.classifier.requires_web_search(request):
                return [RoutingTarget.LOCAL_LLM]
            return []
        elif failed_target == RoutingTarget.CODE_REVIEWER:
            return [RoutingTarget.OPENAI_API, RoutingTarget.LOCAL_LLM]
        elif failed_target == RoutingTarget.CODE_GENERATOR:
            return [RoutingTarget.OPENAI_API, RoutingTarget.LOCAL_LLM]
        
        return []
    
    def _generate_routing_reasoning(self, classification: TaskClassification,
                                  original_target: RoutingTarget, final_target: RoutingTarget,
                                  fallback_targets: List[RoutingTarget]) -> str:
        """Generate human-readable reasoning for routing decision."""
        reasoning_parts = []
        
        # Classification-based reasoning
        if classification.requires_web_search:
            reasoning_parts.append("requires web search capabilities")
        if classification.requires_advanced_reasoning:
            reasoning_parts.append("requires advanced reasoning")
        if classification.complexity_score > 0.7:
            reasoning_parts.append(f"high complexity score ({classification.complexity_score:.2f})")
        elif classification.complexity_score < 0.3:
            reasoning_parts.append(f"low complexity score ({classification.complexity_score:.2f})")
        
        # Target adjustment reasoning
        if original_target != final_target:
            reasoning_parts.append(f"adjusted from {original_target.name} to {final_target.name} due to availability")
        
        # Confidence reasoning
        if classification.confidence_score > 0.8:
            reasoning_parts.append("high confidence classification")
        elif classification.confidence_score < 0.5:
            reasoning_parts.append("low confidence classification")
        
        base_reasoning = f"Routed to {final_target.name}"
        if reasoning_parts:
            base_reasoning += f" based on: {', '.join(reasoning_parts)}"
        
        if fallback_targets:
            base_reasoning += f". Fallback options: {', '.join(t.name for t in fallback_targets)}"
        
        return base_reasoning
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics and performance metrics."""
        total_requests = self._routing_stats['total_requests']
        
        stats = {
            'total_requests': total_requests,
            'successful_routes': self._routing_stats['successful_routes'],
            'fallback_routes': self._routing_stats['fallback_routes'],
            'failed_routes': self._routing_stats['failed_routes'],
            'success_rate': (self._routing_stats['successful_routes'] / total_requests * 100) if total_requests > 0 else 0,
            'fallback_rate': (self._routing_stats['fallback_routes'] / total_requests * 100) if total_requests > 0 else 0,
            'target_usage': self._routing_stats['target_usage'].copy(),
            'recent_decisions': len(self.routing_log),
            'log_capacity': self._max_log_entries
        }
        
        # Add target usage percentages
        if total_requests > 0:
            stats['target_usage_percentages'] = {
                target: (count / total_requests * 100) 
                for target, count in self._routing_stats['target_usage'].items()
            }
        
        return stats
    
    def get_recent_routing_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent routing decisions for debugging and analysis."""
        recent_decisions = self.routing_log[-limit:] if self.routing_log else []
        
        return [
            {
                'timestamp': decision.timestamp.isoformat(),
                'target': decision.target.name,
                'reasoning': decision.reasoning,
                'complexity_score': decision.classification.complexity_score,
                'confidence_score': decision.classification.confidence_score,
                'requires_web_search': decision.classification.requires_web_search,
                'requires_advanced_reasoning': decision.classification.requires_advanced_reasoning,
                'fallback_targets': [t.name for t in decision.fallback_targets]
            }
            for decision in recent_decisions
        ]
    
    def clear_routing_log(self) -> None:
        """Clear the routing decision log."""
        self.routing_log.clear()
        self.logger.info("Routing decision log cleared")
    
    def is_healthy(self) -> bool:
        """Check if the central router is healthy and ready to process requests."""
        try:
            # Check if classifier is working
            test_request = UserRequest(content="test", request_type=RequestType.SIMPLE)
            test_classification = self.classifier.classify_task(test_request)
            
            if not test_classification:
                return False
            
            # Check if at least one target is available
            available_targets = [
                self._is_target_available(target) for target in RoutingTarget
            ]
            
            return any(available_targets)
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False