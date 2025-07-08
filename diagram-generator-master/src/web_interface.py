"""
Enhanced Web Interface for Diagram Generator

This module provides a comprehensive FastAPI web interface for the diagram generator 
application with advanced error handling, retry mechanisms, and detailed debugging.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .llm_service import LLMService, LLMRequest
from .diagram_generator import DiagramGenerator, DiagramRequest


class GenerateRequest(BaseModel):
    """Enhanced request model for diagram generation endpoint."""
    description: str
    model: str = "deepseek-coder:6.7b"
    filename: Optional[str] = None
    include_debug: bool = False
    max_retries: int = 3
    auto_cleanup: bool = True


class HealthResponse(BaseModel):
    """Enhanced response model for health check."""
    status: str
    llm_connected: bool
    output_dir_exists: bool
    graphviz_available: bool
    available_models: List[str]
    system_info: Dict[str, Any]


class GenerateResponse(BaseModel):
    """Enhanced response model for diagram generation."""
    success: bool
    python_code: Optional[str] = None
    diagram_path: Optional[str] = None
    diagram_url: Optional[str] = None
    processing_time: Optional[float] = None
    retry_count: Optional[int] = None
    debug_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WebInterface:
    """Enhanced web interface class with comprehensive error handling."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize the enhanced web interface.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Enhanced Diagram Generator",
            description="Convert natural language to cloud architecture diagrams with comprehensive error handling",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
          # Initialize enhanced services
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.llm_service = LLMService(host=ollama_host)
        self.diagram_generator = DiagramGenerator()
        
        # Setup enhanced logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup templates and static files
        self.templates = Jinja2Templates(directory="templates")
        
        # Create directories if they don't exist
        self._ensure_directories()
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.app.mount("/output", StaticFiles(directory="output"), name="output")
        
        # Setup routes
        self._setup_routes()
        
        # Track generation statistics
        self.stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_processing_time": 0.0,
            "retry_statistics": {}
        }
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = ["templates", "static", "output"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Ensured directory exists: {directory}")
    
    def _setup_routes(self):
        """Setup enhanced FastAPI routes with comprehensive error handling."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Serve the enhanced main page."""
            try:
                return self.templates.TemplateResponse("index.html", {
                    "request": request,
                    "stats": self.stats
                })
            except Exception as e:
                self.logger.error(f"Error serving index page: {str(e)}")
                return HTMLResponse(content="<h1>Error loading page</h1>", status_code=500)
        
        @self.app.get("/health", response_model=HealthResponse)
        async def enhanced_health_check():
            """Comprehensive health check endpoint."""
            try:
                start_time = time.time()
                
                # Check LLM connection
                llm_connected = await self.llm_service.check_connection()
                
                # Check available models
                available_models = await self.llm_service.list_available_models()
                
                # Check output directory
                output_dir_exists = Path("output").exists()
                
                # Check Graphviz
                graphviz_available = self.diagram_generator._check_graphviz()
                
                # System info
                system_info = {
                    "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                    "working_directory": str(Path.cwd()),
                    "output_directory": str(Path("output").absolute()),
                    "health_check_time": f"{time.time() - start_time:.3f}s",
                    "generation_stats": self.stats
                }
                
                status = "healthy"
                if not llm_connected:
                    status = "degraded - LLM unavailable"
                elif not graphviz_available:
                    status = "degraded - Graphviz unavailable"
                elif not output_dir_exists:
                    status = "degraded - Output directory issue"
                
                return HealthResponse(
                    status=status,
                    llm_connected=llm_connected,
                    output_dir_exists=output_dir_exists,
                    graphviz_available=graphviz_available,
                    available_models=available_models,
                    system_info=system_info
                )
                
            except Exception as e:
                self.logger.error(f"Health check error: {str(e)}")
                return HealthResponse(
                    status="error",
                    llm_connected=False,
                    output_dir_exists=False,
                    graphviz_available=False,
                    available_models=[],
                    system_info={"error": str(e)}
                )
        
        @self.app.post("/generate", response_model=GenerateResponse)
        async def enhanced_generate_diagram(request: GenerateRequest, background_tasks: BackgroundTasks):
            """Enhanced diagram generation with comprehensive error handling and retries."""
            start_time = time.time()
            self.stats["total_requests"] += 1
            
            try:
                self.logger.info(f"Starting diagram generation: '{request.description[:50]}...'")
                
                # Step 1: Generate Python code using LLM with retries
                llm_response = await self.llm_service.generate_diagram_code(
                    description=request.description,
                    model=request.model,
                    max_retries=request.max_retries
                )
                
                if not llm_response.success:
                    self.stats["failed_generations"] += 1
                    error_detail = f"LLM generation failed: {llm_response.error}"
                    
                    debug_info = None
                    if request.include_debug:
                        debug_info = {
                            "stage": "llm_generation",
                            "llm_raw_response": llm_response.raw_response,
                            "error_details": llm_response.error,
                            "model_used": request.model,
                            "retry_count": llm_response.retry_count,
                            "processing_time": llm_response.processing_time,
                            "max_retries": request.max_retries
                        }
                    
                    return GenerateResponse(
                        success=False,
                        error=error_detail,
                        retry_count=llm_response.retry_count,
                        processing_time=llm_response.processing_time,
                        debug_info=debug_info
                    )
                
                # Step 2: Generate diagram from code with retries
                diagram_response = await self.diagram_generator.generate_diagram(
                    python_code=llm_response.python_code,
                    filename=request.filename,
                    max_retries=request.max_retries
                )
                
                if not diagram_response.success:
                    self.stats["failed_generations"] += 1
                    error_detail = f"Diagram generation failed: {diagram_response.error}"
                    
                    debug_info = None
                    if request.include_debug:
                        debug_info = {
                            "stage": "diagram_generation",
                            "python_code": llm_response.python_code,
                            "llm_raw_response": llm_response.raw_response,
                            "execution_log": diagram_response.execution_log,
                            "validation_details": diagram_response.validation_details,
                            "error_details": diagram_response.error,
                            "model_used": request.model,
                            "llm_retry_count": llm_response.retry_count,
                            "diagram_retry_count": diagram_response.retry_count,
                            "total_processing_time": time.time() - start_time
                        }
                    
                    return GenerateResponse(
                        success=False,
                        error=error_detail,
                        retry_count=diagram_response.retry_count,
                        processing_time=time.time() - start_time,
                        debug_info=debug_info
                    )
                
                # Step 3: Success! Prepare response
                total_processing_time = time.time() - start_time
                self.stats["successful_generations"] += 1
                self._update_stats(total_processing_time, llm_response.retry_count + diagram_response.retry_count)
                
                # Schedule cleanup if requested
                if request.auto_cleanup:
                    background_tasks.add_task(self.diagram_generator.cleanup_old_diagrams, 50)
                
                response_data = GenerateResponse(
                    success=True,
                    python_code=llm_response.python_code,
                    diagram_path=diagram_response.diagram_path,
                    diagram_url=f"/output/{Path(diagram_response.diagram_path).name}",
                    processing_time=total_processing_time,
                    retry_count=llm_response.retry_count + diagram_response.retry_count
                )
                
                # Add comprehensive debug information if requested
                if request.include_debug:
                    response_data.debug_info = {
                        "stage": "success",
                        "llm_raw_response": llm_response.raw_response,
                        "execution_log": diagram_response.execution_log,
                        "validation_details": diagram_response.validation_details,
                        "model_used": request.model,
                        "llm_retry_count": llm_response.retry_count,
                        "llm_processing_time": llm_response.processing_time,
                        "diagram_retry_count": diagram_response.retry_count,
                        "diagram_processing_time": diagram_response.processing_time,
                        "total_processing_time": total_processing_time,
                        "generated_file_info": diagram_response.validation_details,
                        "system_info": {
                            "max_retries_used": request.max_retries,
                            "auto_cleanup": request.auto_cleanup
                        }
                    }
                
                self.logger.info(f"Successfully generated diagram in {total_processing_time:.2f}s")
                return response_data
                
            except Exception as e:
                self.stats["failed_generations"] += 1
                self.logger.error(f"Unexpected error in generate_diagram: {str(e)}")
                
                error_detail = f"Unexpected error: {str(e)}"
                debug_info = None
                
                if request.include_debug:
                    debug_info = {
                        "stage": "unexpected_error",
                        "error_details": str(e),
                        "error_type": type(e).__name__,
                        "model_used": request.model,
                        "total_processing_time": time.time() - start_time,
                        "system_state": {
                            "total_requests": self.stats["total_requests"],
                            "failed_generations": self.stats["failed_generations"]
                        }
                    }
                
                return GenerateResponse(
                    success=False,
                    error=error_detail,
                    processing_time=time.time() - start_time,
                    debug_info=debug_info
                )        
        @self.app.get("/diagrams")
        async def list_diagrams():
            """List all generated diagrams with enhanced metadata."""
            try:
                diagrams = self.diagram_generator.list_diagrams()
                return {
                    "diagrams": diagrams,
                    "total_count": len(diagrams),
                    "total_size_mb": sum(d.get("size_mb", 0) for d in diagrams),
                    "statistics": self.stats
                }
            except Exception as e:
                self.logger.error(f"Error listing diagrams: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error listing diagrams: {str(e)}")
        
        @self.app.get("/download/{filename}")
        async def download_diagram(filename: str):
            """Download a specific diagram file with validation."""
            try:
                file_path = Path("output") / filename
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="File not found")
                
                # Validate file
                validation = self.diagram_generator._validate_diagram_file(file_path)
                if not validation["valid"]:
                    raise HTTPException(status_code=400, detail=f"Invalid file: {validation['error']}")
                
                return FileResponse(
                    path=str(file_path),
                    filename=filename,
                    media_type='application/octet-stream'
                )
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error downloading file: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/diagrams/{filename}")
        async def delete_diagram(filename: str):
            """Delete a specific diagram file with validation."""
            try:
                file_path = Path("output") / filename
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="File not found")
                
                file_size = file_path.stat().st_size
                file_path.unlink()
                
                self.logger.info(f"Deleted diagram: {filename} ({file_size} bytes)")
                return {
                    "success": True, 
                    "message": f"File {filename} deleted",
                    "file_size": file_size
                }
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error deleting file: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/cleanup")
        async def cleanup_diagrams(max_files: int = 50):
            """Clean up old diagram files."""
            try:
                removed_count = self.diagram_generator.cleanup_old_diagrams(max_files)
                return {
                    "success": True,
                    "removed_count": removed_count,
                    "max_files": max_files
                }
            except Exception as e:
                self.logger.error(f"Error during cleanup: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stats")
        async def get_statistics():
            """Get generation statistics."""
            return {
                "statistics": self.stats,
                "diagrams_info": {
                    "total_files": len(self.diagram_generator.list_diagrams()),
                    "output_directory": str(Path("output").absolute())
                }
            }
        
        @self.app.post("/test-llm")
        async def test_llm_connection(model: str = "deepseek-coder:6.7b"):
            """Test LLM connection with a simple query."""
            try:
                test_response = await self.llm_service.generate_diagram_code(
                    description="A simple web server connected to a database",
                    model=model,
                    max_retries=1
                )
                
                return {
                    "success": test_response.success,
                    "model": model,
                    "response_time": test_response.processing_time,
                    "retry_count": test_response.retry_count,
                    "has_code": bool(test_response.python_code),
                    "error": test_response.error if not test_response.success else None
                }
            except Exception as e:
                return {
                    "success": False,
                    "model": model,
                    "error": str(e)
                }
    
    def _update_stats(self, processing_time: float, retry_count: int):
        """Update generation statistics."""
        # Update average processing time
        total_successful = self.stats["successful_generations"]
        current_avg = self.stats["average_processing_time"]
        self.stats["average_processing_time"] = (
            (current_avg * (total_successful - 1) + processing_time) / total_successful
        )
        
        # Update retry statistics
        retry_key = str(retry_count)
        if retry_key not in self.stats["retry_statistics"]:
            self.stats["retry_statistics"][retry_key] = 0
        self.stats["retry_statistics"][retry_key] += 1
    
    def run(self):
        """Run the enhanced web interface."""
        import uvicorn
        
        self.logger.info(f"Starting enhanced diagram generator server on {self.host}:{self.port}")
        self.logger.info(f"API documentation available at: http://{self.host}:{self.port}/docs")
        
        uvicorn.run(
            self.app, 
            host=self.host, 
            port=self.port,
            log_level="info",
            reload=False  # Disable reload in production
        )
