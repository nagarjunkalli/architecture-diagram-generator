"""
Startup Script for Development

This script helps set up and run the diagram generator in development mode.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def check_graphviz():
    """Check if Graphviz is installed."""
    try:
        subprocess.run(['dot', '-V'], capture_output=True, check=True)
        print("✅ Graphviz is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Graphviz is not installed")
        print("Please install Graphviz:")
        print("  Windows: choco install graphviz  OR  winget install graphviz")
        print("  macOS:   brew install graphviz")
        print("  Linux:   sudo apt-get install graphviz")
        return False


def check_ollama():
    """Check if Ollama is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
        else:
            print("❌ Ollama is not responding properly")
            return False
    except Exception:
        print("❌ Ollama is not running")
        print("Please start Ollama:")
        print("  Option 1: docker-compose up -d ollama")
        print("  Option 2: Install Ollama locally and run 'ollama serve'")
        return False


def setup_environment():
    """Set up the development environment."""
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("✅ Created .env file from .env.example")
        else:
            env_content = """OLLAMA_HOST=http://localhost:11434
DEBUG=true
LOG_LEVEL=INFO"""
            env_file.write_text(env_content)
            print("✅ Created default .env file")


def main():
    """Main setup and run function."""
    print("🚀 Setting up Diagram Generator...")
    print()
    
    # Check prerequisites
    graphviz_ok = check_graphviz()
    print()
    
    # Setup environment
    setup_environment()
    print()
    
    # Check Ollama
    ollama_ok = check_ollama()
    print()
    
    if not graphviz_ok:
        print("⚠️  Graphviz is required for diagram generation")
        
    if not ollama_ok:
        print("⚠️  Ollama is required for LLM functionality")
        print("You can still run the app, but diagram generation will fail without Ollama")
    
    print()
    print("Starting the application...")
    
    # Start the main application
    try:
        from main import main as app_main
        app_main()
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Error starting application: {e}")


if __name__ == "__main__":
    main()
