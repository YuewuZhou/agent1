"""
Demo script showing basic usage of the Multi-Agent Routing System.
"""

from datetime import datetime
from .models import UserRequest, RequestType, SystemConfig
from .utils import setup_logging, get_logger, ConfigManager
from .core import CentralRouter


def main():
    """Demonstrate basic system functionality."""
    # Initialize configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Setup logging
    setup_logging(config.logging_config)
    logger = get_logger(__name__)
    
    logger.info("Multi-Agent Routing System Demo Starting")
    
    # Create sample requests
    requests = [
        UserRequest(
            content="What is the capital of France?",
            request_type=RequestType.SIMPLE,
            user_id="demo_user"
        ),
        UserRequest(
            content="Please review this Python code for security issues: def login(user, password): return user == 'admin' and password == '123'",
            request_type=RequestType.CODE_REVIEW,
            user_id="demo_user"
        ),
        UserRequest(
            content="Generate a Python function to calculate fibonacci numbers",
            request_type=RequestType.CODE_GENERATION,
            user_id="demo_user"
        )
    ]
    
    # Initialize router
    router = CentralRouter()
    
    # Process requests
    for i, request in enumerate(requests, 1):
        logger.info(f"Processing request {i}: {request.content[:50]}...")
        response = router.route_request(request)
        logger.info(f"Response {i}: {response.content}")
        print(f"\nRequest {i}: {request.content}")
        print(f"Response: {response.content}")
        print(f"Source: {response.source_agent}")
        print("-" * 50)
    
    logger.info("Demo completed successfully")


if __name__ == "__main__":
    main()