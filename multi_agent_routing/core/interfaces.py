"""
Interface implementations for Local LLM and OpenAI API.
"""

import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import requests
import json
import openai
from openai import OpenAI
import random

from ..models import UserRequest, AgentResponse, ConversationContext
from ..models.config import LocalLLMConfig, OpenAIConfig, RateLimitConfig
from ..models.enums import RoutingTarget, APIStatus
from ..utils import get_logger
from ..utils.error_handling import APIError, ResourceUnavailableError, handle_error, retry_with_backoff


class APIResponse:
    """Response from external API."""
    def __init__(self, content: str, success: bool = True, 
                 error_message: Optional[str] = None, cost: float = 0.0,
                 tokens_used: int = 0, model: str = ""):
        self.content = content
        self.success = success
        self.error_message = error_message
        self.cost = cost
        self.tokens_used = tokens_used
        self.model = model
        self.timestamp = datetime.now()


class SearchResults:
    """Results from web search."""
    def __init__(self, results: List[Dict[str, Any]], query: str):
        self.results = results
        self.query = query
        self.timestamp = datetime.now()
        self.count = len(results)


class ResourceStatus:
    """Status information for a resource."""
    def __init__(self, available: bool = True, response_time: float = 0.0, 
                 error_message: Optional[str] = None):
        self.available = available
        self.response_time = response_time
        self.error_message = error_message
        self.last_check = datetime.now()


class LocalLLMInterface:
    """
    Interface for local LLM resources using Ollama.
    
    Handles model loading, context management, and response generation
    with timeout handling and performance monitoring.
    """
    
    def __init__(self, config: Optional[LocalLLMConfig] = None):
        self.config = config or LocalLLMConfig()
        self.logger = get_logger(__name__)
        self._conversations: Dict[str, ConversationContext] = {}
        self._conversation_lock = threading.Lock()
        self._model_loaded = False
        self._last_availability_check = None
        self._availability_status = ResourceStatus()
        
        # Performance tracking
        self._response_times = []
        self._max_response_time_samples = 100
        
        self.logger.info(f"Initialized LocalLLMInterface with model: {self.config.model_path}")
    
    def initialize_model(self) -> bool:
        """
        Initialize and load the local LLM model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            self.logger.info(f"Initializing model: {self.config.model_path}")
            
            # Check if Ollama is running
            response = requests.get(f"{self.config.base_url}/api/tags", 
                                  timeout=self.config.timeout_seconds)
            
            if response.status_code != 200:
                self.logger.error(f"Ollama server not accessible: {response.status_code}")
                return False
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.config.model_path not in model_names:
                self.logger.warning(f"Model {self.config.model_path} not found. Available models: {model_names}")
                # Try to pull the model
                self._pull_model()
            
            # Test model with a simple prompt
            test_response = self._make_ollama_request("Hello", timeout=10)
            if test_response:
                self._model_loaded = True
                self.logger.info(f"Model {self.config.model_path} initialized successfully")
                return True
            else:
                self.logger.error("Model initialization test failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            return False
    
    def _pull_model(self) -> bool:
        """Pull the model if it's not available locally."""
        try:
            self.logger.info(f"Pulling model: {self.config.model_path}")
            response = requests.post(
                f"{self.config.base_url}/api/pull",
                json={"name": self.config.model_path},
                timeout=300  # 5 minutes for model download
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to pull model: {str(e)}")
            return False
    
    def generate_response(self, request: UserRequest) -> AgentResponse:
        """
        Generate response using local LLM with timeout handling.
        
        Args:
            request: User request to process
            
        Returns:
            AgentResponse: Generated response with metadata
        """
        start_time = time.time()
        
        try:
            # Check if model is initialized
            if not self._model_loaded and not self.initialize_model():
                return AgentResponse(
                    content="Local LLM model not available",
                    source_agent="LocalLLM",
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Model initialization failed"
                )
            
            # Get conversation context
            context = self.manage_context(request.context.conversation_id if request.context else "default")
            
            # Prepare prompt with context
            prompt = self._prepare_prompt_with_context(request.content, context)
            
            # Generate response
            response_content = self._make_ollama_request(prompt, self.config.timeout_seconds)
            
            if response_content is None:
                return AgentResponse(
                    content="Failed to generate response",
                    source_agent="LocalLLM",
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="Request timeout or generation failed"
                )
            
            processing_time = time.time() - start_time
            
            # Update conversation context
            if request.context:
                self._update_conversation_context(context, request.content, response_content)
            
            # Track performance
            self._track_response_time(processing_time)
            
            self.logger.info(f"Generated response in {processing_time:.2f}s")
            
            return AgentResponse(
                content=response_content,
                source_agent="LocalLLM",
                processing_time=processing_time,
                metadata={
                    "model": self.config.model_path,
                    "context_length": len(prompt),
                    "temperature": self.config.temperature
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error generating response: {str(e)}")
            
            return AgentResponse(
                content="Error processing request",
                source_agent="LocalLLM",
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _make_ollama_request(self, prompt: str, timeout: int) -> Optional[str]:
        """Make a request to Ollama API."""
        try:
            payload = {
                "model": self.config.model_path,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_ctx": self.config.max_context_length
                }
            }
            
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout after {timeout} seconds")
            return None
        except Exception as e:
            self.logger.error(f"Request failed: {str(e)}")
            return None
    
    def _prepare_prompt_with_context(self, content: str, context: ConversationContext) -> str:
        """Prepare prompt with conversation context."""
        if not context.history:
            return content
        
        # Build context from recent history (limit to avoid context overflow)
        context_parts = []
        total_length = len(content)
        
        for entry in reversed(context.history[-10:]):  # Last 10 exchanges
            entry_text = f"Human: {entry['human']}\nAssistant: {entry['assistant']}\n"
            if total_length + len(entry_text) > self.config.max_context_length * 0.8:
                break
            context_parts.insert(0, entry_text)
            total_length += len(entry_text)
        
        if context_parts:
            return "\n".join(context_parts) + f"\nHuman: {content}\nAssistant:"
        else:
            return f"Human: {content}\nAssistant:"
    
    def manage_context(self, conversation_id: str) -> ConversationContext:
        """
        Manage conversation context for multi-turn interactions.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            ConversationContext: Context object for the conversation
        """
        with self._conversation_lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = ConversationContext(
                    conversation_id=conversation_id
                )
            return self._conversations[conversation_id]
    
    def _update_conversation_context(self, context: ConversationContext, 
                                   human_message: str, assistant_message: str):
        """Update conversation context with new exchange."""
        with self._conversation_lock:
            context.history.append({
                "human": human_message,
                "assistant": assistant_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Limit history size to prevent memory issues
            if len(context.history) > 50:
                context.history = context.history[-25:]  # Keep last 25 exchanges
    
    def check_availability(self) -> ResourceStatus:
        """
        Check if Local LLM resources are available.
        
        Returns:
            ResourceStatus: Current availability status
        """
        try:
            start_time = time.time()
            
            # Check Ollama server
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                response_time = time.time() - start_time
                self._availability_status = ResourceStatus(
                    available=True,
                    response_time=response_time
                )
            else:
                self._availability_status = ResourceStatus(
                    available=False,
                    error_message=f"Server returned {response.status_code}"
                )
                
        except Exception as e:
            self._availability_status = ResourceStatus(
                available=False,
                error_message=str(e)
            )
        
        self._last_availability_check = datetime.now()
        return self._availability_status
    
    def estimate_processing_time(self, request: UserRequest) -> float:
        """
        Estimate processing time for a request.
        
        Args:
            request: User request to estimate
            
        Returns:
            float: Estimated processing time in seconds
        """
        # Base estimate on content length and historical performance
        content_length = len(request.content)
        base_time = max(1.0, content_length / 1000)  # 1 second per 1000 characters minimum
        
        # Adjust based on historical performance
        if self._response_times:
            avg_response_time = sum(self._response_times) / len(self._response_times)
            base_time = max(base_time, avg_response_time * 0.8)
        
        # Add context overhead if applicable
        if request.context and request.context.conversation_id in self._conversations:
            context = self._conversations[request.context.conversation_id]
            if context.history:
                base_time *= 1.2  # 20% overhead for context processing
        
        return min(base_time, self.config.timeout_seconds)
    
    def _track_response_time(self, response_time: float):
        """Track response time for performance monitoring."""
        self._response_times.append(response_time)
        if len(self._response_times) > self._max_response_time_samples:
            self._response_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._response_times:
            return {"avg_response_time": 0.0, "sample_count": 0}
        
        return {
            "avg_response_time": sum(self._response_times) / len(self._response_times),
            "min_response_time": min(self._response_times),
            "max_response_time": max(self._response_times),
            "sample_count": len(self._response_times),
            "availability": self._availability_status.available
        }
    
    def enforce_response_time_limits(self, request: UserRequest) -> bool:
        """
        Check if request can be processed within time limits.
        
        Args:
            request: User request to check
            
        Returns:
            bool: True if request can be processed within limits
        """
        estimated_time = self.estimate_processing_time(request)
        
        # Check against configured timeout
        if estimated_time > self.config.timeout_seconds:
            self.logger.warning(f"Estimated processing time {estimated_time:.2f}s exceeds timeout {self.config.timeout_seconds}s")
            return False
        
        # Check against recent performance trends
        if self._response_times:
            recent_avg = sum(self._response_times[-10:]) / min(len(self._response_times), 10)
            if recent_avg > self.config.timeout_seconds * 0.8:  # 80% of timeout threshold
                self.logger.warning(f"Recent average response time {recent_avg:.2f}s approaching timeout limit")
                return False
        
        return True
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """
        Get current resource utilization metrics.
        
        Returns:
            Dict containing resource utilization information
        """
        return {
            "active_conversations": len(self._conversations),
            "model_loaded": self._model_loaded,
            "last_availability_check": self._last_availability_check.isoformat() if self._last_availability_check else None,
            "availability_status": {
                "available": self._availability_status.available,
                "response_time": self._availability_status.response_time,
                "error_message": self._availability_status.error_message,
                "last_check": self._availability_status.last_check.isoformat()
            },
            "config": {
                "model_path": self.config.model_path,
                "timeout_seconds": self.config.timeout_seconds,
                "max_context_length": self.config.max_context_length,
                "max_concurrent_requests": self.config.max_concurrent_requests
            }
        }
    
    def is_healthy(self) -> bool:
        """
        Perform a comprehensive health check.
        
        Returns:
            bool: True if the interface is healthy and ready to process requests
        """
        try:
            # Check basic availability
            status = self.check_availability()
            if not status.available:
                return False
            
            # Check if response times are reasonable
            if self._response_times:
                recent_avg = sum(self._response_times[-5:]) / min(len(self._response_times), 5)
                if recent_avg > self.config.timeout_seconds:
                    self.logger.warning(f"Recent response times too high: {recent_avg:.2f}s")
                    return False
            
            # Check model status
            if not self._model_loaded:
                self.logger.warning("Model not loaded")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False


class OpenAIInterface:
    """
    Interface for OpenAI API with rate limiting, cost tracking, and web search integration.
    
    Handles API communication, quota management, error handling, and cost estimation
    with comprehensive retry logic and fallback mechanisms.
    """
    
    def __init__(self, config: Optional[OpenAIConfig] = None):
        self.config = config or OpenAIConfig()
        self.logger = get_logger(__name__)
        
        # Initialize OpenAI client
        if not self.config.api_key:
            self.logger.warning("OpenAI API key not provided")
            self._client = None
        else:
            self._client = OpenAI(api_key=self.config.api_key)
        
        # Rate limiting tracking
        self._request_times: List[datetime] = []
        self._rate_limit_lock = threading.Lock()
        
        # Cost tracking
        self._daily_cost = 0.0
        self._monthly_cost = 0.0
        self._last_cost_reset = datetime.now()
        self._cost_lock = threading.Lock()
        
        # API status tracking
        self._api_status = APIStatus.ACTIVE
        self._last_status_check = None
        self._consecutive_failures = 0
        
        # Performance tracking
        self._response_times = []
        self._max_response_time_samples = 100
        
        self.logger.info("Initialized OpenAIInterface")
    
    def send_request(self, prompt: str, model: str = None, 
                    parameters: Dict[str, Any] = None) -> APIResponse:
        """
        Send request to OpenAI API with rate limiting and error handling.
        
        Args:
            prompt: The prompt to send to the API
            model: Model to use (defaults to config preference)
            parameters: Additional parameters for the API call
            
        Returns:
            APIResponse: Response from the API with metadata
        """
        start_time = time.time()
        
        try:
            # Check API availability
            if not self._client:
                return APIResponse(
                    content="OpenAI API client not initialized",
                    success=False,
                    error_message="API key not provided"
                )
            
            # Check rate limits
            if not self._check_rate_limits():
                return APIResponse(
                    content="Rate limit exceeded",
                    success=False,
                    error_message="API rate limit exceeded, please try again later"
                )
            
            # Check cost limits
            if not self._check_cost_limits():
                return APIResponse(
                    content="Cost limit exceeded",
                    success=False,
                    error_message="Daily or monthly cost limit exceeded"
                )
            
            # Determine model to use
            if not model:
                model = self.config.model_preferences.get("simple", "gpt-3.5-turbo")
            
            # Prepare parameters
            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": parameters.get("temperature", 0.7) if parameters else 0.7,
                "max_tokens": parameters.get("max_tokens", 1000) if parameters else 1000
            }
            
            # Make API request with retry logic
            response = self._make_api_request_with_retry(api_params)
            
            if response.success:
                # Track successful request
                self._track_request()
                self._track_cost(response.cost)
                self._track_response_time(time.time() - start_time)
                self._consecutive_failures = 0
                self._api_status = APIStatus.ACTIVE
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3:
                    self._api_status = APIStatus.ERROR
            
            return response
            
        except Exception as e:
            self.logger.error(f"Unexpected error in send_request: {str(e)}")
            self._consecutive_failures += 1
            
            return APIResponse(
                content="Internal error occurred",
                success=False,
                error_message=str(e)
            )
    
    def _make_api_request_with_retry(self, api_params: Dict[str, Any]) -> APIResponse:
        """Make API request with exponential backoff retry logic."""
        max_retries = self.config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                # Make the actual API call
                response = self._client.chat.completions.create(**api_params)
                
                # Extract response data
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 0
                cost = self._calculate_cost(tokens_used, api_params["model"])
                
                return APIResponse(
                    content=content,
                    success=True,
                    cost=cost,
                    tokens_used=tokens_used,
                    model=api_params["model"]
                )
                
            except openai.RateLimitError as e:
                self.logger.warning(f"Rate limit hit on attempt {attempt + 1}: {str(e)}")
                self._api_status = APIStatus.RATE_LIMITED
                
                if attempt < max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    self.logger.info(f"Waiting {delay:.2f}s before retry...")
                    time.sleep(delay)
                else:
                    return APIResponse(
                        content="Rate limit exceeded",
                        success=False,
                        error_message="API rate limit exceeded after retries"
                    )
            
            except openai.APIError as e:
                self.logger.error(f"OpenAI API error on attempt {attempt + 1}: {str(e)}")
                
                if attempt < max_retries and "quota" not in str(e).lower():
                    delay = self._calculate_backoff_delay(attempt)
                    time.sleep(delay)
                else:
                    if "quota" in str(e).lower():
                        self._api_status = APIStatus.QUOTA_EXCEEDED
                    return APIResponse(
                        content="API error occurred",
                        success=False,
                        error_message=str(e)
                    )
            
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                
                if attempt < max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    time.sleep(delay)
                else:
                    return APIResponse(
                        content="Request failed",
                        success=False,
                        error_message=str(e)
                    )
        
        return APIResponse(
            content="Max retries exceeded",
            success=False,
            error_message="Request failed after maximum retry attempts"
        )
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        base_delay = self.config.rate_limits.backoff_base_delay
        max_delay = self.config.rate_limits.backoff_max_delay
        
        delay = min(base_delay * (2 ** attempt), max_delay)
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter
    
    def _check_rate_limits(self) -> bool:
        """Check if request is within rate limits."""
        with self._rate_limit_lock:
            now = datetime.now()
            
            # Clean old requests (older than 1 hour)
            self._request_times = [
                req_time for req_time in self._request_times
                if now - req_time < timedelta(hours=1)
            ]
            
            # Check per-minute limit
            recent_requests = [
                req_time for req_time in self._request_times
                if now - req_time < timedelta(minutes=1)
            ]
            
            if len(recent_requests) >= self.config.rate_limits.requests_per_minute:
                self.logger.warning("Per-minute rate limit exceeded")
                return False
            
            # Check per-hour limit
            if len(self._request_times) >= self.config.rate_limits.requests_per_hour:
                self.logger.warning("Per-hour rate limit exceeded")
                return False
            
            return True
    
    def _track_request(self):
        """Track a successful request for rate limiting."""
        with self._rate_limit_lock:
            self._request_times.append(datetime.now())
    
    def _check_cost_limits(self) -> bool:
        """Check if request is within cost limits."""
        with self._cost_lock:
            self._update_cost_tracking()
            
            if self._daily_cost >= self.config.cost_limits.daily_limit_usd:
                self.logger.warning(f"Daily cost limit exceeded: ${self._daily_cost:.2f}")
                return False
            
            if self._monthly_cost >= self.config.cost_limits.monthly_limit_usd:
                self.logger.warning(f"Monthly cost limit exceeded: ${self._monthly_cost:.2f}")
                return False
            
            return True
    
    def _track_cost(self, cost: float):
        """Track API usage cost."""
        with self._cost_lock:
            self._daily_cost += cost
            self._monthly_cost += cost
            
            # Check alert thresholds
            daily_threshold = self.config.cost_limits.daily_limit_usd * (self.config.cost_limits.alert_threshold_percent / 100)
            monthly_threshold = self.config.cost_limits.monthly_limit_usd * (self.config.cost_limits.alert_threshold_percent / 100)
            
            if self._daily_cost >= daily_threshold:
                self.logger.warning(f"Daily cost approaching limit: ${self._daily_cost:.2f} / ${self.config.cost_limits.daily_limit_usd:.2f}")
            
            if self._monthly_cost >= monthly_threshold:
                self.logger.warning(f"Monthly cost approaching limit: ${self._monthly_cost:.2f} / ${self.config.cost_limits.monthly_limit_usd:.2f}")
    
    def _update_cost_tracking(self):
        """Update cost tracking periods."""
        now = datetime.now()
        
        # Reset daily cost if it's a new day
        if now.date() > self._last_cost_reset.date():
            self._daily_cost = 0.0
        
        # Reset monthly cost if it's a new month
        if now.month != self._last_cost_reset.month or now.year != self._last_cost_reset.year:
            self._monthly_cost = 0.0
        
        self._last_cost_reset = now
    
    def _calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost based on token usage and model."""
        # Simplified cost calculation - in practice, this would use actual OpenAI pricing
        cost_per_token = self.config.cost_limits.cost_per_token
        
        # Different models have different costs
        model_multipliers = {
            "gpt-4": 10.0,
            "gpt-4-turbo": 5.0,
            "gpt-3.5-turbo": 1.0
        }
        
        multiplier = 1.0
        for model_name, mult in model_multipliers.items():
            if model_name in model.lower():
                multiplier = mult
                break
        
        return tokens * cost_per_token * multiplier
    
    def handle_rate_limiting(self) -> None:
        """Handle rate limiting by implementing delays and backoff."""
        if self._api_status == APIStatus.RATE_LIMITED:
            # Calculate appropriate delay based on recent request patterns
            with self._rate_limit_lock:
                if self._request_times:
                    last_request = max(self._request_times)
                    time_since_last = (datetime.now() - last_request).total_seconds()
                    
                    # If we made a request very recently, wait a bit
                    if time_since_last < 1.0:
                        delay = 1.0 - time_since_last
                        self.logger.info(f"Rate limiting: waiting {delay:.2f}s")
                        time.sleep(delay)
    
    def perform_web_search(self, query: str) -> SearchResults:
        """
        Perform web search using OpenAI's capabilities.
        
        Note: This is a simplified implementation. In practice, you might use
        OpenAI's function calling with a web search tool or integrate with
        a dedicated search API.
        
        Args:
            query: Search query
            
        Returns:
            SearchResults: Search results with metadata
        """
        try:
            # Use OpenAI to generate a response that simulates web search
            # In a real implementation, this would integrate with actual search APIs
            search_prompt = f"""
            You are a web search assistant. For the query "{query}", provide a structured response 
            that includes relevant information as if you had access to current web search results.
            Format your response as if you found 3-5 relevant sources with titles, URLs, and summaries.
            """
            
            response = self.send_request(
                prompt=search_prompt,
                model=self.config.model_preferences.get("complex", "gpt-4"),
                parameters={"temperature": 0.3, "max_tokens": 1500}
            )
            
            if response.success:
                # Parse the response into structured search results
                # This is a simplified parsing - in practice, you'd use more sophisticated methods
                results = self._parse_search_response(response.content, query)
                return SearchResults(results, query)
            else:
                self.logger.error(f"Web search failed: {response.error_message}")
                return SearchResults([], query)
                
        except Exception as e:
            self.logger.error(f"Error performing web search: {str(e)}")
            return SearchResults([], query)
    
    def _parse_search_response(self, content: str, query: str) -> List[Dict[str, Any]]:
        """Parse OpenAI response into structured search results."""
        # Simplified parsing - in practice, this would be more sophisticated
        results = []
        
        # Split content into potential result sections
        sections = content.split('\n\n')
        
        for i, section in enumerate(sections[:5]):  # Limit to 5 results
            if len(section.strip()) > 50:  # Only consider substantial sections
                results.append({
                    "title": f"Search Result {i+1}",
                    "url": f"https://example.com/result-{i+1}",
                    "summary": section.strip()[:200] + "..." if len(section) > 200 else section.strip(),
                    "relevance_score": 0.8 - (i * 0.1)  # Decreasing relevance
                })
        
        return results
    
    def check_api_status(self) -> APIStatus:
        """
        Check current API status.
        
        Returns:
            APIStatus: Current status of the API
        """
        try:
            # Perform a lightweight test request
            test_response = self.send_request(
                prompt="Hello",
                parameters={"max_tokens": 5}
            )
            
            if test_response.success:
                self._api_status = APIStatus.ACTIVE
                self._consecutive_failures = 0
            else:
                if "rate limit" in test_response.error_message.lower():
                    self._api_status = APIStatus.RATE_LIMITED
                elif "quota" in test_response.error_message.lower():
                    self._api_status = APIStatus.QUOTA_EXCEEDED
                else:
                    self._api_status = APIStatus.ERROR
            
            self._last_status_check = datetime.now()
            return self._api_status
            
        except Exception as e:
            self.logger.error(f"Error checking API status: {str(e)}")
            self._api_status = APIStatus.UNAVAILABLE
            return self._api_status
    
    def estimate_cost(self, request: UserRequest) -> float:
        """
        Estimate cost for processing a request.
        
        Args:
            request: User request to estimate cost for
            
        Returns:
            float: Estimated cost in USD
        """
        # Estimate tokens based on content length
        # Rough approximation: 1 token ≈ 4 characters
        estimated_tokens = len(request.content) // 4
        
        # Add overhead for response (assume response is similar length to input)
        estimated_tokens *= 2
        
        # Determine model based on request type
        model = "gpt-3.5-turbo"  # Default
        if hasattr(request, 'request_type'):
            if request.request_type.name in ["COMPLEX", "CODE_REVIEW", "CODE_GENERATION"]:
                model = self.config.model_preferences.get("complex", "gpt-4")
        
        return self._calculate_cost(estimated_tokens, model)
    
    def generate_response(self, request: UserRequest) -> AgentResponse:
        """
        Generate response using OpenAI API.
        
        Args:
            request: User request to process
            
        Returns:
            AgentResponse: Generated response with metadata
        """
        start_time = time.time()
        
        try:
            # Determine appropriate model based on request type
            model = self._select_model_for_request(request)
            
            # Prepare prompt with context if available
            prompt = self._prepare_prompt_with_context(request)
            
            # Estimate cost before making request
            estimated_cost = self.estimate_cost(request)
            
            # Make API request
            api_response = self.send_request(
                prompt=prompt,
                model=model,
                parameters={
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            )
            
            processing_time = time.time() - start_time
            
            if api_response.success:
                self.logger.info(f"Generated response in {processing_time:.2f}s, cost: ${api_response.cost:.4f}")
                
                return AgentResponse(
                    content=api_response.content,
                    source_agent="OpenAI",
                    processing_time=processing_time,
                    cost_estimate=api_response.cost,
                    metadata={
                        "model": api_response.model,
                        "tokens_used": api_response.tokens_used,
                        "estimated_cost": estimated_cost,
                        "actual_cost": api_response.cost
                    }
                )
            else:
                return AgentResponse(
                    content=f"Failed to generate response: {api_response.error_message}",
                    source_agent="OpenAI",
                    processing_time=processing_time,
                    success=False,
                    error_message=api_response.error_message
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error generating response: {str(e)}")
            
            return AgentResponse(
                content="Error processing request",
                source_agent="OpenAI",
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _select_model_for_request(self, request: UserRequest) -> str:
        """Select appropriate model based on request characteristics."""
        if hasattr(request, 'request_type'):
            request_type = request.request_type.name
            
            if request_type in ["CODE_REVIEW", "CODE_GENERATION", "COMPLEX"]:
                return self.config.model_preferences.get("code", "gpt-4")
            elif request_type == "WEB_SEARCH":
                return self.config.model_preferences.get("complex", "gpt-4")
            else:
                return self.config.model_preferences.get("simple", "gpt-3.5-turbo")
        
        # Default to simple model
        return self.config.model_preferences.get("simple", "gpt-3.5-turbo")
    
    def _prepare_prompt_with_context(self, request: UserRequest) -> str:
        """Prepare prompt with conversation context if available."""
        if request.context and request.context.history:
            # Build context from conversation history
            context_parts = []
            
            for entry in request.context.history[-5:]:  # Last 5 exchanges
                if 'human' in entry and 'assistant' in entry:
                    context_parts.append(f"Human: {entry['human']}")
                    context_parts.append(f"Assistant: {entry['assistant']}")
            
            if context_parts:
                context_str = "\n".join(context_parts)
                return f"{context_str}\nHuman: {request.content}\nAssistant:"
        
        return request.content
    
    def _track_response_time(self, response_time: float):
        """Track response time for performance monitoring."""
        self._response_times.append(response_time)
        if len(self._response_times) > self._max_response_time_samples:
            self._response_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance and usage statistics."""
        with self._cost_lock:
            self._update_cost_tracking()
            
            stats = {
                "api_status": self._api_status.name,
                "consecutive_failures": self._consecutive_failures,
                "daily_cost": self._daily_cost,
                "monthly_cost": self._monthly_cost,
                "daily_limit": self.config.cost_limits.daily_limit_usd,
                "monthly_limit": self.config.cost_limits.monthly_limit_usd,
                "requests_last_hour": len(self._request_times),
                "rate_limit_per_hour": self.config.rate_limits.requests_per_hour
            }
            
            if self._response_times:
                stats.update({
                    "avg_response_time": sum(self._response_times) / len(self._response_times),
                    "min_response_time": min(self._response_times),
                    "max_response_time": max(self._response_times),
                    "response_time_samples": len(self._response_times)
                })
            
            return stats
    
    def is_healthy(self) -> bool:
        """
        Perform comprehensive health check.
        
        Returns:
            bool: True if the interface is healthy and ready to process requests
        """
        try:
            # Check if client is initialized
            if not self._client:
                return False
            
            # Check API status
            status = self.check_api_status()
            if status in [APIStatus.UNAVAILABLE, APIStatus.ERROR]:
                return False
            
            # Check cost limits
            if not self._check_cost_limits():
                return False
            
            # Check rate limits
            if not self._check_rate_limits():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False
    
    def get_detailed_cost_breakdown(self, request: UserRequest) -> Dict[str, Any]:
        """
        Get detailed cost breakdown for a request.
        
        Args:
            request: User request to analyze
            
        Returns:
            Dict containing detailed cost information
        """
        try:
            # Estimate tokens for input
            input_tokens = len(request.content) // 4  # Rough approximation
            
            # Estimate output tokens based on request type
            output_multiplier = 1.0
            if hasattr(request, 'request_type'):
                if request.request_type.name in ["CODE_GENERATION", "CODE_REVIEW"]:
                    output_multiplier = 2.0  # Code tasks typically need longer responses
                elif request.request_type.name == "COMPLEX":
                    output_multiplier = 1.5
            
            output_tokens = int(input_tokens * output_multiplier)
            total_tokens = input_tokens + output_tokens
            
            # Determine model and get pricing
            model = self._select_model_for_request(request)
            cost_per_token = self._get_model_cost_per_token(model)
            
            estimated_cost = total_tokens * cost_per_token
            
            return {
                "input_tokens": input_tokens,
                "estimated_output_tokens": output_tokens,
                "total_estimated_tokens": total_tokens,
                "model": model,
                "cost_per_token": cost_per_token,
                "estimated_cost_usd": estimated_cost,
                "daily_remaining_budget": max(0, self.config.cost_limits.daily_limit_usd - self._daily_cost),
                "monthly_remaining_budget": max(0, self.config.cost_limits.monthly_limit_usd - self._monthly_cost),
                "can_afford": estimated_cost <= (self.config.cost_limits.daily_limit_usd - self._daily_cost)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating cost breakdown: {str(e)}")
            return {
                "error": str(e),
                "estimated_cost_usd": 0.0,
                "can_afford": False
            }
    
    def _get_model_cost_per_token(self, model: str) -> float:
        """Get cost per token for a specific model."""
        # Simplified pricing - in practice, this would use actual OpenAI pricing
        base_cost = self.config.cost_limits.cost_per_token
        
        model_multipliers = {
            "gpt-4": 15.0,      # More expensive for GPT-4
            "gpt-4-turbo": 7.5,  # Mid-range pricing
            "gpt-3.5-turbo": 1.0  # Base pricing
        }
        
        for model_name, multiplier in model_multipliers.items():
            if model_name in model.lower():
                return base_cost * multiplier
        
        return base_cost  # Default pricing
    
    def handle_api_failure(self, error: Exception, context: Dict[str, Any] = None) -> AgentResponse:
        """
        Handle API failures with comprehensive error analysis and recovery suggestions.
        
        Args:
            error: The exception that occurred
            context: Additional context about the failure
            
        Returns:
            AgentResponse with error details and recovery suggestions
        """
        try:
            context = context or {}
            
            # Analyze the error type
            if isinstance(error, openai.RateLimitError):
                error_type = "rate_limit"
                message = "API rate limit exceeded. Please try again in a few minutes."
                recovery_suggestions = [
                    "Wait for rate limit to reset",
                    "Consider upgrading API plan",
                    "Use local LLM as fallback"
                ]
            elif isinstance(error, openai.AuthenticationError):
                error_type = "authentication"
                message = "API authentication failed. Please check your API key."
                recovery_suggestions = [
                    "Verify API key is correct",
                    "Check API key permissions",
                    "Regenerate API key if necessary"
                ]
            elif "quota" in str(error).lower():
                error_type = "quota_exceeded"
                message = "API quota exceeded. Please check your usage limits."
                recovery_suggestions = [
                    "Check current usage in OpenAI dashboard",
                    "Upgrade API plan if needed",
                    "Wait for quota reset"
                ]
            elif isinstance(error, (ConnectionError, TimeoutError)):
                error_type = "connection"
                message = "Connection to OpenAI API failed."
                recovery_suggestions = [
                    "Check internet connection",
                    "Retry request",
                    "Use local LLM as fallback"
                ]
            else:
                error_type = "unknown"
                message = f"Unexpected API error: {str(error)}"
                recovery_suggestions = [
                    "Retry request",
                    "Check API status",
                    "Contact support if issue persists"
                ]
            
            # Log detailed error information
            self.logger.error(f"API failure - Type: {error_type}, Error: {str(error)}, Context: {context}")
            
            # Update failure tracking
            self._consecutive_failures += 1
            
            # Create comprehensive error response
            error_response = AgentResponse(
                content=message,
                source_agent="OpenAI",
                processing_time=0.0,
                success=False,
                error_message=str(error),
                metadata={
                    "error_type": error_type,
                    "recovery_suggestions": recovery_suggestions,
                    "consecutive_failures": self._consecutive_failures,
                    "context": context,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return error_response
            
        except Exception as e:
            # Fallback error handling
            self.logger.error(f"Error in error handler: {str(e)}")
            return AgentResponse(
                content="Critical error in API error handling",
                source_agent="OpenAI",
                processing_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def validate_request_before_processing(self, request: UserRequest) -> Dict[str, Any]:
        """
        Validate request before sending to API to prevent unnecessary costs.
        
        Args:
            request: User request to validate
            
        Returns:
            Dict with validation results and recommendations
        """
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "cost_estimate": 0.0,
            "recommendations": []
        }
        
        try:
            # Check content length
            if len(request.content) > 50000:  # Arbitrary large content threshold
                validation_result["warnings"].append("Request content is very large, may result in high costs")
                validation_result["recommendations"].append("Consider breaking into smaller requests")
            
            if len(request.content.strip()) == 0:
                validation_result["valid"] = False
                validation_result["errors"].append("Request content is empty")
                return validation_result
            
            # Check cost limits
            cost_breakdown = self.get_detailed_cost_breakdown(request)
            validation_result["cost_estimate"] = cost_breakdown.get("estimated_cost_usd", 0.0)
            
            if not cost_breakdown.get("can_afford", False):
                validation_result["valid"] = False
                validation_result["errors"].append("Request would exceed daily cost limit")
                validation_result["recommendations"].append("Wait for daily limit reset or increase budget")
            
            # Check API health
            if not self.is_healthy():
                validation_result["valid"] = False
                validation_result["errors"].append("OpenAI API is currently unavailable")
                validation_result["recommendations"].append("Try local LLM or wait for API recovery")
            
            # Check rate limits
            if not self._check_rate_limits():
                validation_result["valid"] = False
                validation_result["errors"].append("Rate limit would be exceeded")
                validation_result["recommendations"].append("Wait before making request")
            
            # Content-based recommendations
            if hasattr(request, 'request_type'):
                if request.request_type.name == "SIMPLE" and len(request.content) < 1000:
                    validation_result["recommendations"].append("Consider using local LLM for simple, short requests")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating request: {str(e)}")
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
            return validation_result
    
    def get_cost_optimization_suggestions(self) -> List[str]:
        """
        Get suggestions for optimizing API costs based on usage patterns.
        
        Returns:
            List of cost optimization suggestions
        """
        suggestions = []
        
        try:
            # Analyze current usage
            stats = self.get_performance_stats()
            
            # Daily cost analysis
            daily_usage_percent = (self._daily_cost / self.config.cost_limits.daily_limit_usd) * 100
            if daily_usage_percent > 80:
                suggestions.append("Daily cost approaching limit - consider using local LLM for simple tasks")
            
            # Monthly cost analysis
            monthly_usage_percent = (self._monthly_cost / self.config.cost_limits.monthly_limit_usd) * 100
            if monthly_usage_percent > 70:
                suggestions.append("Monthly cost high - review usage patterns and consider plan optimization")
            
            # Request frequency analysis
            if len(self._request_times) > self.config.rate_limits.requests_per_hour * 0.8:
                suggestions.append("High request frequency - consider batching requests or using local LLM")
            
            # Model usage suggestions
            if self._consecutive_failures > 0:
                suggestions.append("Recent failures detected - consider implementing fallback to local LLM")
            
            # Response time analysis
            if self._response_times and len(self._response_times) > 10:
                avg_time = sum(self._response_times) / len(self._response_times)
                if avg_time > 10.0:  # 10 seconds
                    suggestions.append("Slow response times - consider using faster models or local LLM for simple tasks")
            
            # General optimization suggestions
            suggestions.extend([
                "Use gpt-3.5-turbo for simple tasks to reduce costs",
                "Implement request caching for repeated queries",
                "Set appropriate max_tokens limits to control costs",
                "Monitor usage regularly to avoid unexpected charges"
            ])
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating cost optimization suggestions: {str(e)}")
            return ["Error analyzing usage patterns for optimization suggestions"]