"""
Test Script for Diagram Generator

This script tests the core functionality of the diagram generator.
"""

import asyncio
from src.llm_service import LLMService
from src.diagram_generator import DiagramGenerator


async def test_llm_service():
    """Test the LLM service."""
    print("🧪 Testing LLM Service...")
    
    llm = LLMService()
    
    # Check connection
    connected = await llm.check_connection()
    print(f"LLM Connection: {'✅' if connected else '❌'}")
    
    if connected:
        # Test code generation
        response = await llm.generate_diagram_code(
            "Create a simple web application with a load balancer and web server"
        )
        
        if response.success:
            print("✅ Code generation successful")
            print("Generated code preview:")
            print(response.python_code[:200] + "..." if len(response.python_code) > 200 else response.python_code)
            return response.python_code
        else:
            print(f"❌ Code generation failed: {response.error}")
    
    return None


async def test_diagram_generator(python_code: str = None):
    """Test the diagram generator."""
    print("\n🎨 Testing Diagram Generator...")
    
    if not python_code:
        # Use a simple test code
        python_code = '''
from diagrams import Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.network import ELB

with Diagram("Test Architecture", show=False, filename="output/test"):
    lb = ELB("Load Balancer")
    web = EC2("Web Server")
    
    lb >> web
'''
    
    generator = DiagramGenerator()
    
    # Test diagram generation
    response = await generator.generate_diagram(python_code, "test-diagram")
    
    if response.success:
        print("✅ Diagram generation successful")
        print(f"Diagram saved to: {response.diagram_path}")
    else:
        print(f"❌ Diagram generation failed: {response.error}")
        if response.execution_log:
            print(f"Execution log: {response.execution_log}")


async def main():
    """Main test function."""
    print("🚀 Running Diagram Generator Tests\n")
    
    # Test LLM service
    generated_code = await test_llm_service()
    
    # Test diagram generator
    await test_diagram_generator(generated_code)
    
    print("\n✨ Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
