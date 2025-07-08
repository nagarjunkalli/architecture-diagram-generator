# Diagram Generator App

An intelligent application that converts natural language descriptions into cloud architecture diagrams using the Python diagrams library and Ollama LLM.

## Features

- Natural language to diagram conversion
- Support for multiple cloud providers (AWS, Azure, GCP, Kubernetes)
- Docker Ollama integration for local LLM processing
- Web interface for easy interaction
- Automatic diagram generation and export

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Graphviz (for diagram rendering)

### Installing Graphviz

**Windows:**
```bash
# Using Chocolatey
choco install graphviz

# Using winget
winget install graphviz
```

**macOS:**
```bash
brew install graphviz
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install graphviz
```

## Quick Start

### Option 1: Development Mode (Recommended for testing)

1. **Install Graphviz:**
   ```bash
   # Windows (using Chocolatey)
   choco install graphviz
   
   # Windows (using winget)
   winget install graphviz
   
   # macOS
   brew install graphviz
   
   # Linux (Ubuntu/Debian)
   sudo apt-get install graphviz
   ```

2. **Install Ollama locally:**
   - Download from: https://ollama.ai
   - Run: `ollama serve` (in a separate terminal)
   - Pull a model: `ollama pull llama3.1:8b`

3. **Run the application:**
   ```bash
   python run_dev.py
   ```

### Option 2: Docker (Full production setup)

1. **Start with Docker Compose:**
   ```bash
   docker-compose up -d ollama
   ```

2. **Pull a model (first time only):**
   ```bash
   docker exec -it diag_gen_ollama_1 ollama pull llama3.1:8b
   ```

3. **Start the application:**
   ```bash
   docker-compose up app
   ```

### Testing the Setup

Run the test script to verify everything works:
```bash
python test_app.py
```

## Usage

1. Open the web interface
2. Enter a natural language description of your architecture
3. The app will generate Python code using the diagrams library
4. View and download the generated diagram

### Example Descriptions

- "Create a web application with load balancer, web servers, and database"
- "Design a microservices architecture on AWS with API Gateway, Lambda functions, and RDS"
- "Show a Kubernetes deployment with ingress, services, and pods"

## Project Structure

```
diag_gen/
├── src/
│   ├── __init__.py
│   ├── llm_service.py      # Ollama LLM integration
│   ├── diagram_generator.py # Diagrams library wrapper
│   └── web_interface.py    # FastAPI web server
├── templates/
│   └── index.html          # Web UI
├── static/
│   └── style.css          # CSS styles
├── output/                 # Generated diagrams
├── docker-compose.yml      # Docker configuration
├── Dockerfile             # Application container
├── requirements.txt        # Python dependencies
├── main.py                # Application entry point
└── README.md              # This file
```

## Development

### VS Code Tasks

The project includes several VS Code tasks (Ctrl+Shift+P → "Tasks: Run Task"):

- **Install Dependencies**: Install Python packages
- **Run Development Server**: Start the app with development setup
- **Test Application**: Run the test suite
- **Start Ollama Docker**: Start Ollama in Docker
- **Stop Docker Services**: Stop all Docker services

### Debugging

Use F5 to start debugging with these configurations:
- **Debug Diagram Generator**: Debug the main application
- **Debug Development Server**: Debug with development setup
- **Run Tests**: Debug the test suite

### Project Structure

```
diag_gen/
├── src/                    # Source code
│   ├── __init__.py
│   ├── llm_service.py      # Ollama LLM integration
│   ├── diagram_generator.py # Diagrams library wrapper
│   └── web_interface.py    # FastAPI web server
├── templates/              # HTML templates
│   └── index.html          # Main web interface
├── static/                 # Static assets
│   └── style.css          # CSS styles
├── output/                 # Generated diagrams
├── .vscode/               # VS Code configuration
│   ├── tasks.json         # Build tasks
│   └── launch.json        # Debug configurations
├── .github/               # GitHub configuration
│   └── copilot-instructions.md # Copilot guidance
├── docker-compose.yml     # Docker services
├── Dockerfile            # App container
├── requirements.txt      # Python dependencies
├── main.py              # Application entry point
├── run_dev.py           # Development runner
├── test_app.py          # Test suite
└── README.md           # This file
```

## Features Implemented

✅ **Natural Language Processing**: Converts plain English to Python diagrams code using Ollama LLM  
✅ **Code Generation**: Generates valid Python code using the diagrams library  
✅ **Security Validation**: Validates generated code for safety before execution  
✅ **Multi-Cloud Support**: AWS, Azure, GCP, Kubernetes diagrams  
✅ **Web Interface**: Modern, responsive web UI with real-time feedback  
✅ **Docker Integration**: Easy deployment with Docker Compose  
✅ **Development Tools**: VS Code tasks, debugging configurations, and test suite  
✅ **Error Handling**: Comprehensive error handling and logging  
✅ **Environment Configuration**: Flexible configuration via environment variables  

## Architecture

The application follows a modular architecture:

1. **Web Layer** (`web_interface.py`): FastAPI server handling HTTP requests
2. **LLM Layer** (`llm_service.py`): Integrates with Ollama for code generation
3. **Execution Layer** (`diagram_generator.py`): Safely executes Python code and generates diagrams
4. **Storage Layer**: File system storage for generated diagrams

## Next Steps

Possible enhancements:
- Add user authentication and project management
- Support for custom diagram templates
- Integration with cloud storage (AWS S3, Azure Blob)
- Real-time collaboration features
- API documentation with Swagger/OpenAPI
- Automated testing with GitHub Actions

## License

MIT License
