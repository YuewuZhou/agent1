"""
Task Classifier implementation for analyzing request complexity and requirements.
"""

import re
import math
from typing import Dict, List, Set, Tuple
from ..models import UserRequest, TaskClassification, RoutingTarget, RequestType
from ..utils import get_logger, ClassificationError


class TaskClassifier:
    """
    Task classifier for analyzing request complexity and requirements.
    
    Analyzes user requests to determine:
    - Complexity score based on content length, keywords, and patterns
    - Web search requirements using keyword matching and intent analysis
    - Advanced reasoning requirements for multi-step problems
    - Confidence scoring and classification metadata
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Web search keywords and patterns
        self.web_search_keywords = {
            'current', 'latest', 'recent', 'today', 'now', 'weather', 'news',
            'price', 'stock', 'market', 'trending', 'happening', 'update',
            'real-time', 'live', 'current events', 'breaking', 'status'
        }
        
        self.web_search_patterns = [
            r'\bwhat\'?s\s+(happening|new|trending|the\s+weather)\b',
            r'\bcurrent\s+(price|status|weather|news)\b',
            r'\blatest\s+(news|update|version)\b',
            r'\btoday\'?s\s+\w+',
            r'\bright\s+now\b',
            r'\breal[- ]?time\b'
        ]
        
        # Advanced reasoning keywords and patterns
        self.reasoning_keywords = {
            'analyze', 'compare', 'evaluate', 'explain', 'why', 'how',
            'strategy', 'plan', 'design', 'architecture', 'algorithm',
            'optimize', 'solve', 'calculate', 'derive', 'prove',
            'reasoning', 'logic', 'inference', 'conclusion', 'hypothesis'
        }
        
        self.reasoning_patterns = [
            r'\bstep[- ]?by[- ]?step\b',
            r'\bmulti[- ]?step\b',
            r'\bcomplex\s+\w+\s+(problem|analysis|solution)\b',
            r'\bwhy\s+\w+.*\?\s*.*\bhow\b',
            r'\bcompare\s+\w+\s+(with|to|and)\s+\w+',
            r'\banalyze\s+\w+\s+(and|then|to)\b'
        ]
        
        # Complexity scoring weights
        self.complexity_weights = {
            'length': 0.3,
            'keywords': 0.25,
            'patterns': 0.25,
            'structure': 0.2
        }
    
    def classify_task(self, request: UserRequest) -> TaskClassification:
        """
        Classify a user request and determine routing requirements.
        
        Args:
            request: The user request to classify
            
        Returns:
            TaskClassification with analysis results
            
        Raises:
            ClassificationError: If classification fails
        """
        try:
            self.logger.info(f"Classifying task: {request.content[:50]}...")
            
            # Calculate complexity score
            complexity_score = self.calculate_complexity_score(request.content)
            
            # Check for web search requirements
            requires_web_search = self.requires_web_search(request)
            
            # Check for advanced reasoning requirements
            requires_advanced_reasoning = self.requires_advanced_reasoning(request)
            
            # Estimate processing time
            estimated_time = self._estimate_processing_time(
                complexity_score, requires_web_search, requires_advanced_reasoning
            )
            
            # Determine recommended target
            recommended_target = self._determine_routing_target(
                complexity_score, requires_web_search, requires_advanced_reasoning, request.request_type
            )
            
            # Calculate confidence score and generate reasoning
            confidence_score, reasoning, metadata = self._calculate_confidence_and_metadata(
                request, complexity_score, requires_web_search, requires_advanced_reasoning
            )
            
            classification = TaskClassification(
                complexity_score=complexity_score,
                requires_web_search=requires_web_search,
                requires_advanced_reasoning=requires_advanced_reasoning,
                estimated_processing_time=estimated_time,
                recommended_target=recommended_target,
                confidence_score=confidence_score,
                reasoning=reasoning,
                metadata=metadata
            )
            
            self.logger.info(f"Classification complete: {recommended_target.name} (confidence: {confidence_score:.2f})")
            return classification
            
        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            raise ClassificationError(f"Failed to classify request: {str(e)}")
    
    def calculate_complexity_score(self, content: str) -> float:
        """
        Calculate complexity score based on content analysis.
        
        Args:
            content: The request content to analyze
            
        Returns:
            Complexity score between 0.0 and 1.0
        """
        if not content or not content.strip():
            return 0.0
        
        # Length-based complexity (normalized)
        length_score = min(len(content) / 1000.0, 1.0)
        
        # Keyword-based complexity
        keyword_score = self._calculate_keyword_complexity(content)
        
        # Pattern-based complexity
        pattern_score = self._calculate_pattern_complexity(content)
        
        # Structure-based complexity (sentences, questions, etc.)
        structure_score = self._calculate_structure_complexity(content)
        
        # Weighted combination
        complexity_score = (
            self.complexity_weights['length'] * length_score +
            self.complexity_weights['keywords'] * keyword_score +
            self.complexity_weights['patterns'] * pattern_score +
            self.complexity_weights['structure'] * structure_score
        )
        
        return min(max(complexity_score, 0.0), 1.0)
    
    def requires_web_search(self, request: UserRequest) -> bool:
        """
        Determine if the request requires web search capabilities.
        
        Args:
            request: The user request to analyze
            
        Returns:
            True if web search is required, False otherwise
        """
        content_lower = request.content.lower()
        
        # Check for web search keywords
        keyword_match = any(keyword in content_lower for keyword in self.web_search_keywords)
        
        # Check for web search patterns
        pattern_match = any(re.search(pattern, content_lower) for pattern in self.web_search_patterns)
        
        # Check request type
        type_match = request.request_type == RequestType.WEB_SEARCH
        
        return keyword_match or pattern_match or type_match
    
    def requires_advanced_reasoning(self, request: UserRequest) -> bool:
        """
        Determine if the request requires advanced reasoning capabilities.
        
        Args:
            request: The user request to analyze
            
        Returns:
            True if advanced reasoning is required, False otherwise
        """
        content_lower = request.content.lower()
        
        # Check for reasoning keywords
        keyword_match = any(keyword in content_lower for keyword in self.reasoning_keywords)
        
        # Check for reasoning patterns
        pattern_match = any(re.search(pattern, content_lower) for pattern in self.reasoning_patterns)
        
        # Check for multiple questions or complex structure
        question_count = content_lower.count('?')
        complex_structure = question_count > 1 or len(content_lower.split('.')) > 3
        
        # Check request type
        type_match = request.request_type == RequestType.COMPLEX
        
        return keyword_match or pattern_match or complex_structure or type_match
    
    def update_classification_model(self, feedback: Dict) -> None:
        """
        Update classification model based on feedback.
        
        Args:
            feedback: Classification feedback for model improvement
        """
        # Placeholder for future machine learning integration
        self.logger.info("Classification model update requested - feature not yet implemented")
    
    def _calculate_keyword_complexity(self, content: str) -> float:
        """Calculate complexity based on keyword presence."""
        content_lower = content.lower()
        
        # Technical keywords that indicate complexity
        technical_keywords = {
            'algorithm', 'architecture', 'design', 'implement', 'optimize',
            'analyze', 'evaluate', 'compare', 'integrate', 'refactor',
            'database', 'api', 'framework', 'library', 'deployment'
        }
        
        matches = sum(1 for keyword in technical_keywords if keyword in content_lower)
        return min(matches / 5.0, 1.0)  # Normalize to 0-1
    
    def _calculate_pattern_complexity(self, content: str) -> float:
        """Calculate complexity based on linguistic patterns."""
        content_lower = content.lower()
        
        # Complex patterns
        complex_patterns = [
            r'\bif\s+.*\bthen\s+.*\belse\b',  # Conditional logic
            r'\bfor\s+each\s+.*\bdo\b',       # Iteration
            r'\bgiven\s+.*\bwhen\s+.*\bthen\b',  # Scenario patterns
            r'\bstep\s+\d+.*step\s+\d+',      # Multi-step processes
            r'\bfirst.*second.*third\b',       # Sequential processes
        ]
        
        matches = sum(1 for pattern in complex_patterns if re.search(pattern, content_lower))
        return min(matches / 3.0, 1.0)  # Normalize to 0-1
    
    def _calculate_structure_complexity(self, content: str) -> float:
        """Calculate complexity based on content structure."""
        sentences = content.split('.')
        questions = content.count('?')
        words = len(content.split())
        
        # Structure complexity factors
        sentence_complexity = min(len(sentences) / 5.0, 1.0)
        question_complexity = min(questions / 3.0, 1.0)
        word_complexity = min(words / 100.0, 1.0)
        
        return (sentence_complexity + question_complexity + word_complexity) / 3.0
    
    def _estimate_processing_time(self, complexity: float, web_search: bool, reasoning: bool) -> float:
        """Estimate processing time based on classification factors."""
        base_time = 5.0  # Base processing time in seconds
        
        # Add time based on complexity
        complexity_time = complexity * 15.0
        
        # Add time for web search
        web_search_time = 10.0 if web_search else 0.0
        
        # Add time for advanced reasoning
        reasoning_time = 8.0 if reasoning else 0.0
        
        return base_time + complexity_time + web_search_time + reasoning_time
    
    def _determine_routing_target(self, complexity: float, web_search: bool, 
                                reasoning: bool, request_type: RequestType) -> RoutingTarget:
        """Determine the optimal routing target based on classification."""
        # Direct routing for specific request types
        if request_type == RequestType.CODE_REVIEW:
            return RoutingTarget.CODE_REVIEWER
        elif request_type == RequestType.CODE_GENERATION:
            return RoutingTarget.CODE_GENERATOR
        elif request_type == RequestType.WEB_SEARCH or web_search:
            return RoutingTarget.OPENAI_API
        
        # Route complex tasks or advanced reasoning to OpenAI API
        if complexity > 0.7 or reasoning:
            return RoutingTarget.OPENAI_API
        
        # Route simple tasks to Local LLM
        return RoutingTarget.LOCAL_LLM
    
    def _calculate_confidence_and_metadata(self, request: UserRequest, complexity: float,
                                         web_search: bool, reasoning: bool) -> Tuple[float, str, Dict]:
        """Calculate confidence score and generate reasoning with metadata."""
        confidence_factors = []
        reasoning_parts = []
        metadata = {
            'content_length': len(request.content),
            'word_count': len(request.content.split()),
            'question_count': request.content.count('?'),
            'analysis_timestamp': request.timestamp.isoformat()
        }
        
        # Confidence based on clear indicators
        if web_search:
            confidence_factors.append(0.9)
            reasoning_parts.append("clear web search indicators detected")
        
        if reasoning:
            confidence_factors.append(0.85)
            reasoning_parts.append("advanced reasoning patterns identified")
        
        if request.request_type in [RequestType.CODE_REVIEW, RequestType.CODE_GENERATION]:
            confidence_factors.append(0.95)
            reasoning_parts.append("explicit request type specified")
        
        # Confidence based on complexity clarity
        if complexity < 0.3:
            confidence_factors.append(0.8)
            reasoning_parts.append("low complexity indicators")
        elif complexity > 0.7:
            confidence_factors.append(0.85)
            reasoning_parts.append("high complexity indicators")
        else:
            confidence_factors.append(0.6)
            reasoning_parts.append("moderate complexity with mixed indicators")
        
        # Calculate overall confidence
        if confidence_factors:
            confidence_score = sum(confidence_factors) / len(confidence_factors)
        else:
            confidence_score = 0.5  # Default moderate confidence
        
        # Generate reasoning text
        if reasoning_parts:
            reasoning_text = f"Classification based on: {', '.join(reasoning_parts)}"
        else:
            reasoning_text = "Classification based on general content analysis"
        
        # Add detailed metadata
        metadata.update({
            'complexity_breakdown': {
                'length_factor': min(len(request.content) / 1000.0, 1.0),
                'keyword_matches': self._count_keyword_matches(request.content),
                'pattern_matches': self._count_pattern_matches(request.content)
            },
            'classification_factors': {
                'web_search_required': web_search,
                'advanced_reasoning_required': reasoning,
                'complexity_score': complexity
            }
        })
        
        return min(max(confidence_score, 0.0), 1.0), reasoning_text, metadata
    
    def _count_keyword_matches(self, content: str) -> Dict[str, int]:
        """Count matches for different keyword categories."""
        content_lower = content.lower()
        return {
            'web_search_keywords': sum(1 for kw in self.web_search_keywords if kw in content_lower),
            'reasoning_keywords': sum(1 for kw in self.reasoning_keywords if kw in content_lower)
        }
    
    def _count_pattern_matches(self, content: str) -> Dict[str, int]:
        """Count matches for different pattern categories."""
        content_lower = content.lower()
        return {
            'web_search_patterns': sum(1 for pattern in self.web_search_patterns 
                                     if re.search(pattern, content_lower)),
            'reasoning_patterns': sum(1 for pattern in self.reasoning_patterns 
                                    if re.search(pattern, content_lower))
        }