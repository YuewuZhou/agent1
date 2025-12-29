# Multi-Agent Routing System

A coordinator-based multi-agent system that intelligently routes tasks between local LLM resources and OpenAI API based on task complexity and requirements.

## Project Structure

```
multi_agent_routing/
├── __init__.py                 # Main package exports
├── models/                     # Data models and configuration
│   ├── __init__.py
│   ├── core.py                # Core data models (UserRequest, AgentResponse, etc.)
│   ├── config.py              # Configuration models
│   └── enums.py               # Enumerations (RequestType, RoutingTarget, etc.)
├── core/                      # Core system components
│   ├── __init__.py
│   ├── router.py              # Central Router (placeholder)
│   ├── classifier.py          # Task Classifier (placeholder)
│   ├── interfaces.py          # LLM interfaces (placeholder)
│   └── agents.py              # Specialized agents (placeholder)
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── logging.py             # Logging framework
│   ├── config_manager.py      # Configuration management
│   └── error_handling.py      # Error handling and custom exceptions
├── demo.py                    # Demo script
└── README.md                  # This file
```

## Features Implemented (Task 1)

✅ **Project Structure**: Complete Python package structure with proper modules  
✅ **Core Data Models**: UserRequest, TaskClassification, RoutingDecision, AgentResponse  
✅ **Configuration Management**: SystemConfig, LocalLLMConfig, OpenAIConfig with validation  
✅ **Logging Framework**: Structured logging with file rotation and console output  
✅ **Error Handling**: Custom exceptions and retry mechanisms with exponential backoff  

## Quick Start

```python
from multi_agent_routing.models import UserRequest, RequestType, SystemConfig
from multi_agent_routing.utils import setup_logging, ConfigManager
from multi_agent_routing.core import CentralRouter

# Initialize configuration
config_manager = ConfigManager()
config = config_manager.load_config()

# Setup logging
setup_logging(config.logging_config)

# Create a request
request = UserRequest(
    content="What is the capital of France?",
    request_type=RequestType.SIMPLE,
    user_id="demo_user"
)

# Route the request (placeholder implementation)
router = CentralRouter()
response = router.route_request(request)
print(f"Response: {response.content}")
```

## Running the Demo

```bash
python -m multi_agent_routing.demo
```

## Running Tests

```bash
python test_basic_functionality.py
```

## Configuration

The system uses a JSON configuration file (`config.json`) that is automatically created with default values. You can customize:

- Local LLM settings (model path, timeout, context length)
- OpenAI API configuration (API key, model preferences, rate limits)
- Routing thresholds (complexity, confidence scores)
- Logging configuration (level, file rotation, format)

## Next Steps

The following components are placeholders and will be implemented in subsequent tasks:

- **Task Classifier** (Task 2): Intelligent task complexity analysis
- **Local LLM Interface** (Task 3): Integration with local language models
- **OpenAI API Interface** (Task 4): External API communication
- **Central Router** (Task 6): Intelligent routing logic with fallback mechanisms
- **Code Reviewer Agent** (Task 7): Specialized code analysis
- **Code Generator Agent** (Task 8): Code generation and modification

## Requirements

See `requirements.txt` for the complete list of dependencies. Key requirements:
- Python 3.8+
- LangChain ecosystem for LLM integration
- OpenAI API client
- Standard Python libraries for logging, configuration, and error handling