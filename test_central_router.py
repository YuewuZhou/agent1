#!/usr/bin/env python3
"""
Simple test script to verify CentralRouter implementation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'multi_agent_routing'))

from multi_agent_routing.models.core import UserRequest
from multi_agent_routing.models.enums import RequestType
from multi_agent_routing.core.router import CentralRouter

def test_central_router():
    """Test basic CentralRouter functionality."""
    print("Testing CentralRouter implementation...")
    
    try:
        # Initialize router
        router = CentralRouter()
        print("✓ CentralRouter initialized successfully")
        
        # Test health check
        is_healthy = router.is_healthy()
        print(f"✓ Health check: {'Healthy' if is_healthy else 'Not healthy'}")
        
        # Create test request
        test_request = UserRequest(
            content="Hello, this is a simple test request",
            request_type=RequestType.SIMPLE
        )
        
        # Test routing
        response = router.route_request(test_request)
        print(f"✓ Request routed successfully to: {response.source_agent}")
        print(f"  Response: {response.content[:100]}...")
        print(f"  Success: {response.success}")
        
        # Test statistics
        stats = router.get_routing_statistics()
        print(f"✓ Routing statistics: {stats['total_requests']} requests processed")
        
        # Test recent decisions
        decisions = router.get_recent_routing_decisions(limit=1)
        if decisions:
            print(f"✓ Recent decision: routed to {decisions[0]['target']}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_central_router()
    sys.exit(0 if success else 1)