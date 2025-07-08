"""
LLM Service for Natural Language to Diagram Code Conversion

This module handles communication with Ollama to convert natural language
descriptions into Python code that uses the diagrams library with comprehensive
error handling and retry mechanisms.
"""

import asyncio
import logging
import os
import re
import time
from typing import Dict, Optional, Any, List
import ollama
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMRequest(BaseModel):
    """Request model for LLM service."""
    prompt: str
    model: str = "deepseek-coder:6.7b"
    max_retries: int = 3


class LLMResponse(BaseModel):
    """Response model for LLM service."""
    success: bool
    python_code: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    retry_count: int = 0
    processing_time: Optional[float] = None


class LLMService:
    """Enhanced service class for interacting with Ollama LLM with retry mechanisms."""
    
    def __init__(self, host: str = None):
        """
        Initialize the LLM service.
        
        Args:
            host: Ollama server host URL (defaults to environment variable or localhost)
        """
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=self.host)
        self.logger = logging.getLogger(__name__)
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.backoff_multiplier = 2.0
        
    async def generate_diagram_code(self, description: str, model: str = "deepseek-coder:6.7b", max_retries: int = None) -> LLMResponse:
        """
        Generate Python diagrams code from natural language description with retry logic.
        
        Args:
            description: Natural language description of the architecture
            model: Ollama model to use for generation
            max_retries: Maximum number of retry attempts
            
        Returns:
            LLMResponse with generated code or error information
        """
        start_time = time.time()
        max_retries = max_retries or self.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{max_retries + 1} to generate diagram code")
                
                # Create enhanced prompt with comprehensive documentation
                prompt = self._create_enhanced_prompt(description)
                
                response = self.client.chat(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    options={
                        "temperature": 0.1,  # Lower temperature for more consistent code generation
                        "top_p": 0.9,
                        "num_predict": 2000
                    }
                )
                
                raw_response = response['message']['content']
                python_code = self._extract_python_code(raw_response)
                
                # Validate the generated code
                validation_result = self._validate_generated_code(python_code)
                
                if not validation_result["valid"]:
                    if attempt < max_retries:
                        self.logger.warning(f"Code validation failed on attempt {attempt + 1}: {validation_result['error']}")
                        await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
                        continue
                    else:
                        return LLMResponse(
                            success=False,
                            error=f"Code validation failed after {max_retries + 1} attempts: {validation_result['error']}",
                            raw_response=raw_response,
                            retry_count=attempt,
                            processing_time=time.time() - start_time
                        )
                
                # Success case
                return LLMResponse(
                    success=True,
                    python_code=python_code,
                    raw_response=raw_response,
                    retry_count=attempt,
                    processing_time=time.time() - start_time
                )
                
            except ollama.ResponseError as e:
                self.logger.error(f"Ollama response error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
                    continue
                else:
                    return LLMResponse(
                        success=False,
                        error=f"Ollama response error after {max_retries + 1} attempts: {str(e)}",
                        retry_count=attempt,
                        processing_time=time.time() - start_time
                    )
                    
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
                    continue
                else:
                    return LLMResponse(
                        success=False,
                        error=f"Unexpected error after {max_retries + 1} attempts: {str(e)}",
                        retry_count=attempt,
                        processing_time=time.time() - start_time
                    )
          # This should never be reached, but included for completeness
        return LLMResponse(
            success=False,
            error="Maximum retry attempts exceeded",
            retry_count=max_retries,
            processing_time=time.time() - start_time
        )

    def _get_system_prompt(self) -> str:
        """
        Get the comprehensive system prompt with diagrams library documentation.
        
        Returns:
            System prompt for the LLM
        """
        return """You are an expert Python developer specializing in cloud architecture diagrams using the 'diagrams' library.

CRITICAL REQUIREMENTS:
1. Generate ONLY valid Python code using the diagrams library
2. ALWAYS use 'show=False' in Diagram constructor 
3. ALWAYS save to 'output/' directory with filename parameter
4. Handle all imports properly
5. Use meaningful names for all components
6. Follow proper Python syntax and indentation

DIAGRAMS LIBRARY COMPREHENSIVE REFERENCE:

## Core Components

### Diagram Class
- Primary object representing a diagram context
- Use `with Diagram()` context manager for all diagrams
- First parameter becomes the output filename (spaces become underscores)
- Parameters:
  - `show=False`: Disable automatic file opening (ALWAYS USE THIS)
  - `outformat`: png (default), jpg, svg, pdf, dot - can be a list for multiple outputs
  - `filename`: Custom output filename (without extension) - use "output/name" format
  - `direction`: TB, BT, LR (default), RL for data flow direction
  - `graph_attr`, `node_attr`, `edge_attr`: Custom Graphviz attributes

### Node Types by Provider
- **AWS**: `diagrams.aws.compute.EC2`, `diagrams.aws.database.RDS`, `diagrams.aws.network.ELB`, `diagrams.aws.storage.S3`, `diagrams.aws.integration.SQS`, `diagrams.aws.compute.Lambda`
- **Azure**: `diagrams.azure.compute.FunctionApps`, `diagrams.azure.storage.BlobStorage`, `diagrams.azure.compute.VirtualMachines`
- **GCP**: `diagrams.gcp.compute.AppEngine`, `diagrams.gcp.ml.AutoML`, `diagrams.gcp.database.SQL`, `diagrams.gcp.compute.GKE`
- **Kubernetes**: `diagrams.k8s.compute.Pod`, `diagrams.k8s.network.Service`, `diagrams.k8s.network.Ingress`, `diagrams.k8s.storage.PV`, `diagrams.k8s.storage.PVC`
- **On-Premises**: `diagrams.onprem.compute.Server`, `diagrams.onprem.database.PostgreSQL`, `diagrams.onprem.network.Nginx`, `diagrams.onprem.monitoring.Prometheus`
- **Oracle Cloud**: `diagrams.oci.compute.VirtualMachine`, `diagrams.oci.network.Firewall`

### Data Flow Operators
- `>>`: Left to right connection
- `<<`: Right to left connection  
- `-`: Undirected connection
- Group connections: `node >> [node1, node2, node3] >> target`
- Use parentheses when mixing `-` with shift operators

### Clusters
- Group related nodes: `with Cluster("DB Cluster"):`
- Support unlimited nesting depth
- Can connect cluster nodes to external nodes

### Edges
- Custom styling: `Edge(color="red", style="dashed", label="custom")`
- Supported styles: solid, dashed, dotted, bold
- Colors: standard color names or hex codes

## Common Patterns

### Basic Web Service
```python
from diagrams import Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB

with Diagram("Web Service", show=False, filename="output/web_service"):
    ELB("load balancer") >> EC2("web server") >> RDS("database")
```

### Microservices with Clustering
```python
from diagrams import Cluster, Diagram
from diagrams.aws.compute import ECS
from diagrams.aws.database import RDS
from diagrams.aws.network import Route53, ELB

with Diagram("Microservices", show=False, filename="output/microservices"):
    dns = Route53("dns")
    
    with Cluster("Web Services"):
        web_lb = ELB("web lb")
        web_services = [ECS("web1"), ECS("web2")]
    
    with Cluster("Database Cluster"):
        db_primary = RDS("primary")
        db_replicas = [RDS("replica1"), RDS("replica2")]
        db_primary - db_replicas
    
    dns >> web_lb >> web_services >> db_primary
```

BEST PRACTICES:
- Always use `show=False` for programmatic generation
- Import only needed components to reduce memory usage
- Use meaningful cluster and node names
- Group similar resources in clusters
- Use proper data flow direction for readability
- Save diagrams to the `output/` directory

Generate clean, well-structured Python code that creates meaningful architecture diagrams."""

    def _create_enhanced_prompt(self, description: str) -> str:
        """
        Create a comprehensive prompt with examples and best practices.
        
        Args:
            description: User's natural language description
            
        Returns:
            Enhanced prompt for the LLM
        """
        return f"""Generate Python code using the diagrams library for this architecture:

"{description}"

CODE STRUCTURE REQUIREMENTS:
```python
from diagrams import Diagram, Cluster, Edge
from diagrams.[provider].[category] import [Components]

with Diagram("Descriptive Name", show=False, filename="output/descriptive_name", direction="LR"):
    # Create nodes with meaningful names
    # Group related components in clusters
    # Connect with proper data flow
```

EXAMPLES FOR REFERENCE:

Basic Web Service:
```python
from diagrams import Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB

with Diagram("Web Service", show=False, filename="output/web_service"):
    ELB("load balancer") >> EC2("web server") >> RDS("database")
```

Microservices with Clustering:
```python
from diagrams import Cluster, Diagram
from diagrams.aws.compute import ECS
from diagrams.aws.database import RDS
from diagrams.aws.network import Route53, ELB

with Diagram("Microservices", show=False, filename="output/microservices"):
    dns = Route53("dns")
    
    with Cluster("Web Services"):
        web_lb = ELB("web lb")
        web_services = [ECS("web1"), ECS("web2")]
    
    with Cluster("Database Cluster"):
        db_primary = RDS("primary")
        db_replicas = [RDS("replica1"), RDS("replica2")]
        db_primary - db_replicas
    
    dns >> web_lb >> web_services >> db_primary
```

Generate code following these patterns. Be specific with component names and use appropriate clustering."""
    
    def _extract_python_code(self, response: str) -> Optional[str]:
        """
        Extract Python code from LLM response with multiple extraction strategies.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted Python code or None
        """
        # Strategy 1: Extract from python code blocks
        python_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(python_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Strategy 2: Extract from plain code blocks
        code_pattern = r'```\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        if matches:
            code = matches[0].strip()
            if self._looks_like_diagrams_code(code):
                return code
        
        # Strategy 3: Extract from inline code (single backticks)
        inline_pattern = r'`([^`]+)`'
        matches = re.findall(inline_pattern, response)
        for match in matches:
            if self._looks_like_diagrams_code(match):
                return match.strip()
        
        # Strategy 4: Check if entire response is code
        if self._looks_like_diagrams_code(response):
            return response.strip()
        
        # Strategy 5: Extract lines that look like Python code
        lines = response.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if any(keyword in line for keyword in ['from diagrams', 'import', 'with Diagram']):
                in_code_block = True
                code_lines.append(line)
            elif in_code_block and (line.strip().startswith('  ') or line.strip().startswith('\t') or line.strip() == ''):
                code_lines.append(line)
            elif in_code_block and line.strip() and not line.strip().startswith('#'):
                # Check if this line is still part of the code
                if any(keyword in line for keyword in ['=', '>>', '<<', ')', 'Cluster', 'Edge']):
                    code_lines.append(line)
                else:
                    break
        
        if code_lines:
            extracted_code = '\n'.join(code_lines)
            if self._looks_like_diagrams_code(extracted_code):
                return extracted_code.strip()
        
        return None
    
    def _looks_like_diagrams_code(self, code: str) -> bool:
        """
        Check if the code looks like valid diagrams library code.
        
        Args:
            code: Code to check
            
        Returns:
            True if it looks like diagrams code
        """
        required_indicators = [
            any(keyword in code for keyword in ['from diagrams', 'import diagrams']),
            'with Diagram' in code
        ]
        
        optional_indicators = [
            any(provider in code for provider in ['.aws.', '.azure.', '.gcp.', '.k8s.', '.onprem.']),
            any(operator in code for operator in ['>>', '<<', ' - ']),
            'show=False' in code or 'show = False' in code
        ]
        
        return all(required_indicators) and any(optional_indicators)
    
    def _validate_generated_code(self, code: Optional[str]) -> Dict[str, Any]:
        """
        Validate the generated Python code for completeness and correctness.
        
        Args:
            code: Generated Python code
            
        Returns:
            Dictionary with validation result
        """
        if not code:
            return {"valid": False, "error": "No code generated"}
        
        # Check for required imports
        if not any(pattern in code for pattern in ['from diagrams', 'import diagrams']):
            return {"valid": False, "error": "Missing diagrams library import"}
        
        # Check for Diagram context manager
        if 'with Diagram' not in code:
            return {"valid": False, "error": "Missing Diagram context manager"}
        
        # Check for show=False
        if 'show=False' not in code and 'show = False' not in code:
            return {"valid": False, "error": "Missing show=False parameter (required for web interface)"}
        
        # Check for filename parameter
        if 'filename=' not in code:
            return {"valid": False, "error": "Missing filename parameter for output"}
        
        # Check for proper provider imports
        provider_found = any(provider in code for provider in ['.aws.', '.azure.', '.gcp.', '.k8s.', '.onprem.', '.oci.'])
        if not provider_found:
            return {"valid": False, "error": "No cloud provider components found"}
        
        # Check for basic syntax errors
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return {"valid": False, "error": f"Syntax error: {str(e)}"}
          # Check for security issues
        dangerous_patterns = [
            "import os", "import subprocess", "import sys", 
            "exec(", "eval(", "__import__", "open(",
            "file(", "input(", "raw_input("
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return {"valid": False, "error": f"Security risk detected: {pattern}"}
        
        return {"valid": True}
    
    async def check_connection(self) -> bool:
        """
        Check if Ollama service is available with retry logic.
        
        Returns:
            True if connection is successful, False otherwise
        """
        for attempt in range(3):
            try:
                models = self.client.list()
                self.logger.info("Ollama connection successful")
                return True
            except Exception as e:
                self.logger.warning(f"Ollama connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < 2:
                    await asyncio.sleep(1.0)
                
        self.logger.error("Ollama connection failed after 3 attempts")
        return False
    
    async def list_available_models(self) -> List[str]:
        """
        List available models from Ollama.
        
        Returns:
            List of available model names
        """
        try:
            models_response = self.client.list()
            # Handle both possible response formats
            if hasattr(models_response, 'models'):
                return [model.name for model in models_response.models]
            elif isinstance(models_response, dict) and 'models' in models_response:
                return [model.get('name', '') for model in models_response['models']]
            else:
                self.logger.warning(f"Unexpected models response format: {type(models_response)}")
                return []
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return []
