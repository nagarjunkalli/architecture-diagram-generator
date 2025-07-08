"""
Enhanced Diagram Generator Service

This module handles the execution of generated Python code to create
actual diagram images using the diagrams library with comprehensive
error handling, validation, and retry mechanisms based on official
diagrams library documentation.
"""

import asyncio
import ast
import logging
import os
import sys
import tempfile
import traceback
import time
import shutil
import re
from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import subprocess
from pydantic import BaseModel


class DiagramRequest(BaseModel):
    """Request model for diagram generation."""
    python_code: str
    filename: Optional[str] = None
    max_retries: int = 3


class DiagramResponse(BaseModel):
    """Response model for diagram generation."""
    success: bool
    diagram_path: Optional[str] = None
    error: Optional[str] = None
    execution_log: Optional[str] = None
    validation_details: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    processing_time: Optional[float] = None


class DiagramValidator:
    """Validates Python code for diagrams library best practices."""
    
    @staticmethod
    def validate_code(code: str) -> Dict[str, Any]:
        """
        Validate Python code for diagrams library compliance.
        
        Args:
            code: Python code to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        try:
            # Parse the code to AST
            tree = ast.parse(code)
            
            # Check for required patterns
            has_diagram_context = False
            has_show_false = False
            has_output_path = False
            imports = set()
            
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
                        
                # Check for Diagram usage
                elif isinstance(node, ast.With):
                    for item in node.items:
                        if isinstance(item.context_expr, ast.Call):
                            if hasattr(item.context_expr.func, 'id') and item.context_expr.func.id == 'Diagram':
                                has_diagram_context = True
                                
                                # Check for show=False
                                for keyword in item.context_expr.keywords:
                                    if keyword.arg == 'show' and isinstance(keyword.value, ast.Constant):
                                        if keyword.value.value is False:
                                            has_show_false = True
                                    elif keyword.arg == 'filename' and isinstance(keyword.value, ast.Constant):
                                        if 'output/' in keyword.value.value:
                                            has_output_path = True
            
            # Validation checks
            if not has_diagram_context:
                validation_result['errors'].append("Missing 'with Diagram()' context manager")
                validation_result['valid'] = False
                
            if not has_show_false:
                validation_result['warnings'].append("Missing 'show=False' parameter (recommended for programmatic use)")
                
            if not has_output_path:
                validation_result['warnings'].append("Consider using 'filename=\"output/name\"' for organized file structure")
                
            # Check for basic diagrams imports
            if not any('diagrams' in imp for imp in imports):
                validation_result['errors'].append("Missing diagrams library imports")
                validation_result['valid'] = False
                
        except SyntaxError as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Syntax error: {str(e)}")
        except Exception as e:
            validation_result['warnings'].append(f"Validation warning: {str(e)}")
            
        return validation_result


class DiagramGenerator:
    """Enhanced service class for generating diagrams from Python code."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the diagram generator.
        
        Args:
            output_dir: Directory to save generated diagrams
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 2.0  # seconds
        self.backoff_multiplier = 1.5
        
        # Ensure Graphviz is available
        self._check_graphviz()
        
    def _check_graphviz(self) -> bool:
        """
        Check if Graphviz is installed and available.
        
        Returns:
            True if Graphviz is available, False otherwise
        """
        try:
            result = subprocess.run(['dot', '-V'], capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"Graphviz found: {result.stderr.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        self.logger.warning("Graphviz not found. Install it with: https://graphviz.gitlab.io/download/")
        return False
        
    async def generate_diagram(self, python_code: str, filename: Optional[str] = None, max_retries: int = None) -> DiagramResponse:
        """
        Generate a diagram from Python code with comprehensive error handling and retries.
        
        Args:
            python_code: Python code using diagrams library
            filename: Optional custom filename
            max_retries: Maximum number of retry attempts
            
        Returns:
            DiagramResponse with generation results
        """
        start_time = time.time()
        max_retries = max_retries or self.max_retries
        execution_log = []
        
        # Pre-validation
        validation_result = DiagramValidator.validate_code(python_code)
        execution_log.append(f"Code validation: {validation_result}")
        
        if not validation_result['valid']:
            return DiagramResponse(
                success=False,
                error=f"Code validation failed: {'; '.join(validation_result['errors'])}",
                execution_log='\n'.join(execution_log),
                validation_details=validation_result,
                processing_time=time.time() - start_time
            )
        
        # Log warnings if any
        if validation_result['warnings']:
            execution_log.append(f"Validation warnings: {'; '.join(validation_result['warnings'])}")
        
        for attempt in range(max_retries + 1):
            try:
                execution_log.append(f"Attempt {attempt + 1}/{max_retries + 1} to generate diagram")
                
                # Create temporary directory for execution
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Create output directory in temp space
                    temp_output = temp_path / "output"
                    temp_output.mkdir(exist_ok=True)
                    
                    # Prepare the Python code with better error handling
                    enhanced_code = self._enhance_code(python_code, temp_path)
                    execution_log.append(f"Enhanced code prepared")
                    
                    # Write code to temporary file
                    code_file = temp_path / "diagram_code.py"
                    code_file.write_text(enhanced_code, encoding='utf-8')
                    
                    # Execute the code
                    result = await self._execute_code_safely(code_file, temp_path)
                    execution_log.append(f"Code execution result: {result}")
                    
                    if result['success']:
                        # Find generated diagram files
                        diagram_files = list(temp_output.glob("*.png")) + list(temp_output.glob("*.jpg")) + list(temp_output.glob("*.svg"))
                        
                        if diagram_files:
                            # Move the diagram to output directory
                            source_file = diagram_files[0]  # Use first found diagram
                            
                            # Determine final filename
                            if filename:
                                final_filename = f"{filename}.{source_file.suffix.lstrip('.')}"
                            else:
                                final_filename = source_file.name
                                
                            final_path = self.output_dir / final_filename
                            shutil.copy2(source_file, final_path)
                            
                            execution_log.append(f"Diagram saved to: {final_path}")
                            
                            return DiagramResponse(
                                success=True,
                                diagram_path=str(final_path),
                                execution_log='\n'.join(execution_log),
                                validation_details=validation_result,
                                retry_count=attempt,
                                processing_time=time.time() - start_time
                            )
                        else:
                            execution_log.append("No diagram files generated")
                            if attempt < max_retries:
                                await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
                                continue
                    else:
                        execution_log.append(f"Execution failed: {result.get('error', 'Unknown error')}")
                        if attempt < max_retries:
                            await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
                            continue
                            
            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
                execution_log.append(error_msg)
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                
                if attempt < max_retries:
                    await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
                    continue
        
        # All attempts failed
        return DiagramResponse(
            success=False,
            error="All generation attempts failed",
            execution_log='\n'.join(execution_log),
            validation_details=validation_result,
            retry_count=max_retries,
            processing_time=time.time() - start_time        )
    
    def _enhance_code(self, code: str, temp_path: Path) -> str:
        """
        Enhance the Python code with error handling and path fixes.
        
        Args:
            code: Original Python code
            temp_path: Temporary directory path
            
        Returns:
            Enhanced Python code
        """
        # Clean the code first
        cleaned_code = code.strip()
        
        # Remove any markdown code blocks if present
        if cleaned_code.startswith('```python'):
            cleaned_code = cleaned_code[9:]
        if cleaned_code.endswith('```'):
            cleaned_code = cleaned_code[:-3]
        cleaned_code = cleaned_code.strip()
        
        # Add sys.path for imports and working directory
        enhanced_code = f"""import sys
import os
sys.path.insert(0, '{temp_path}')
os.chdir('{temp_path}')

{cleaned_code}
"""
        return enhanced_code
    
    async def _execute_code_safely(self, code_file: Path, working_dir: Path) -> Dict[str, Any]:
        """
        Execute Python code safely in a subprocess.
        
        Args:
            code_file: Path to the Python file to execute
            working_dir: Working directory for execution
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Execute the code in a subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(code_file),
                cwd=str(working_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'stdout': stdout.decode('utf-8', errors='replace'),
                'stderr': stderr.decode('utf-8', errors='replace'),
                'returncode': process.returncode,
                'error': stderr.decode('utf-8', errors='replace') if process.returncode != 0 else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to execute code: {str(e)}",
                'stdout': '',
                'stderr': '',
                'returncode': -1
            }
    
    def _auto_fix_code(self, code: str, error_msg: str) -> Optional[str]:
        """
        Attempt to automatically fix common code issues.
        
        Args:
            code: Original Python code
            error_msg: Error message from execution
            
        Returns:
            Fixed code or None if no fix available
        """
        fixed_code = code
        
        # Fix common issues
        if "show=True" in code or "show=" not in code:
            # Ensure show=False is present
            fixed_code = re.sub(
                r'with\s+Diagram\s*\([^)]*\)',
                lambda m: m.group(0).replace(')', ', show=False)') if 'show=' not in m.group(0) else m.group(0).replace('show=True', 'show=False'),
                fixed_code
            )
        
        if "filename=" not in code:
            # Add filename parameter
            fixed_code = re.sub(
                r'with\s+Diagram\s*\(\s*"([^"]+)"',
                r'with Diagram("\1", filename="output/\1"',
                fixed_code
            )
        
        return fixed_code if fixed_code != code else None

    def cleanup_old_diagrams(self, max_files: int = 50) -> int:
        """
        Clean up old diagram files to prevent disk space issues.
        
        Args:
            max_files: Maximum number of files to keep
            
        Returns:
            Number of files cleaned up
        """
        try:
            output_dir = Path(self.output_dir)
            if not output_dir.exists():
                return 0
            
            # Get all diagram files (png, jpg, svg, pdf)
            diagram_files = []
            for pattern in ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']:
                diagram_files.extend(output_dir.glob(pattern))
            
            # Sort by modification time (newest first)
            diagram_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if len(diagram_files) <= max_files:
                return 0
            
            # Remove oldest files
            files_to_remove = diagram_files[max_files:]
            removed_count = 0
            
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    removed_count += 1
                    self.logger.info(f"Cleaned up old diagram: {file_path.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file_path}: {str(e)}")
            
            return removed_count
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return 0

    def list_diagrams(self) -> List[Dict[str, Any]]:
        """
        List all available diagram files with metadata.
        
        Returns:
            List of diagram info dictionaries
        """
        try:
            output_dir = Path(self.output_dir)
            if not output_dir.exists():
                return []
            
            diagrams = []
            for pattern in ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']:
                for file_path in output_dir.glob(pattern):
                    try:
                        stat = file_path.stat()
                        diagrams.append({
                            'name': file_path.name,
                            'path': str(file_path),
                            'url': f'/output/{file_path.name}',
                            'size': stat.st_size,
                            'created': stat.st_mtime,
                            'extension': file_path.suffix.lower()
                        })
                    except Exception as e:
                        self.logger.warning(f"Error reading file {file_path}: {str(e)}")
            
            # Sort by creation time (newest first)
            diagrams.sort(key=lambda x: x['created'], reverse=True)
            return diagrams
        except Exception as e:
            self.logger.error(f"Error listing diagrams: {str(e)}")
            return []
        

# Global instance for easy access
diagram_generator = DiagramGenerator()


async def generate_diagram_from_code(python_code: str, filename: Optional[str] = None) -> DiagramResponse:
    """
    Convenience function to generate diagrams.
    
    Args:
        python_code: Python code using diagrams library
        filename: Optional custom filename
        
    Returns:
        DiagramResponse with results
    """
    return await diagram_generator.generate_diagram(python_code, filename)
