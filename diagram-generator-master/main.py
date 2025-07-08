"""
Enhanced Main Entry Point for Diagram Generator Application

This module provides the main entry point for the enhanced diagram generator
application with comprehensive error handling and retry mechanisms.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.web_interface import WebInterface


def setup_logging():
    """Setup enhanced logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('diagram_generator.log', mode='a')
        ]
    )


def check_environment():
    """Check if the environment is properly configured."""
    logger = logging.getLogger(__name__)
    
    # Check Python version
    if sys.version_info < (3, 7):
        logger.error("Python 3.7 or higher is required")
        return False
    
    # Check required directories
    required_dirs = ["templates", "static", "output"]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")
      # Check if diagrams library is available
    try:
        import diagrams
        version = getattr(diagrams, '__version__', 'unknown version')
        logger.info(f"Diagrams library found: version {version}")
    except ImportError:
        logger.error("Diagrams library not found. Install with: pip install diagrams")
        return False
    
    # Check if Ollama environment variables are set
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    logger.info(f"Ollama host configured: {ollama_host}")
    
    return True


async def main():
    """Enhanced main function with comprehensive startup checks."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Enhanced Diagram Generator Application")
    logger.info("=" * 60)
    
    # Environment checks
    if not check_environment():
        logger.error("Environment check failed. Exiting.")
        sys.exit(1)
    
    try:
        # Get configuration from environment variables
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        
        logger.info(f"Configuring web interface: {host}:{port}")
        
        # Initialize the enhanced web interface
        web_interface = WebInterface(host=host, port=port)
        
        # Startup health checks
        logger.info("Performing startup health checks...")
        
        # Test LLM connection
        try:
            llm_connected = await web_interface.llm_service.check_connection()
            if llm_connected:
                logger.info("✓ LLM service connection successful")
                
                # List available models
                models = await web_interface.llm_service.list_available_models()
                if models:
                    logger.info(f"✓ Available models: {', '.join(models)}")
                else:
                    logger.warning("⚠ No models found in Ollama")
            else:
                logger.warning("⚠ LLM service connection failed - some features may not work")
        except Exception as e:
            logger.warning(f"⚠ LLM service check failed: {str(e)}")
        
        # Test Graphviz
        if web_interface.diagram_generator._check_graphviz():
            logger.info("✓ Graphviz installation found")
        else:
            logger.warning("⚠ Graphviz not found - diagram generation may fail")
            logger.warning("  Install from: https://graphviz.gitlab.io/download/")
        
        logger.info("=" * 60)
        logger.info("🚀 Enhanced Diagram Generator is starting!")
        logger.info(f"📊 Web interface: http://{host}:{port}")
        logger.info(f"📖 API documentation: http://{host}:{port}/docs")
        logger.info(f"❤️ Health check: http://{host}:{port}/health")
        logger.info("=" * 60)
        
        # Start the web interface - return control to uvicorn
        return web_interface
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


def run_app():
    """Run the application synchronously."""
    import asyncio
    
    # Create a new event loop for the startup process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the async main function to get the web interface
        web_interface = loop.run_until_complete(main())
        
        if web_interface:
            # Close the startup loop
            loop.close()
            
            # Start the web interface with uvicorn (which creates its own event loop)
            web_interface.run()
    finally:
        if not loop.is_closed():
            loop.close()


if __name__ == "__main__":
    try:
        run_app()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
