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
    
    REQUIRED_IMPORTS = {
        'diagrams': ['Diagram', 'Cluster', 'Edge'],
        'aws': ['diagrams.aws.compute', 'diagrams.aws.database', 'diagrams.aws.network', 'diagrams.aws.storage'],
        'azure': ['diagrams.azure.compute', 'diagrams.azure.storage'],
        'gcp': ['diagrams.gcp.compute', 'diagrams.gcp.database', 'diagrams.gcp.ml'],
        'k8s': ['diagrams.k8s.compute', 'diagrams.k8s.network', 'diagrams.k8s.storage'],
        'onprem': ['diagrams.onprem.compute', 'diagrams.onprem.database', 'diagrams.onprem.network'],
    }
    
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
            processing_time=time.time() - start_time
        )
            max_retries: Maximum number of retry attempts
            
        Returns:
            DiagramResponse with result information
        """
        start_time = time.time()
        max_retries = max_retries or self.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Diagram generation attempt {attempt + 1}/{max_retries + 1}")
                
                # Step 1: Enhanced code validation
                validation_result = self._enhanced_validate_code(python_code)
                if not validation_result["valid"]:
                    if attempt < max_retries and validation_result.get("retryable", False):
                        # Try to auto-fix common issues
                        fixed_code = self._auto_fix_code(python_code, validation_result["error"])
                        if fixed_code != python_code:
                            self.logger.info(f"Auto-fixing code issue: {validation_result['error']}")
                            python_code = fixed_code
                            await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
                            continue
                    
                    return DiagramResponse(
                        success=False,
                        error=f"Code validation failed: {validation_result['error']}",
                        validation_details=validation_result,
                        retry_count=attempt,
                        processing_time=time.time() - start_time
                    )
                
                # Step 2: Prepare code for execution
                prepared_code = self._prepare_code_enhanced(python_code, filename)
                
                # Step 3: Execute with comprehensive monitoring
                execution_result = await self._execute_code_enhanced(prepared_code, attempt)
                
                if not execution_result["success"]:
                    if attempt < max_retries and execution_result.get("retryable", False):
                        self.logger.warning(f"Execution failed on attempt {attempt + 1}, retrying: {execution_result['error']}")
                        await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
                        continue
                    
                    return DiagramResponse(
                        success=False,
                        error=execution_result["error"],
                        execution_log=execution_result.get("log"),
                        validation_details=validation_result,
                        retry_count=attempt,
                        processing_time=time.time() - start_time
                    )
                
                # Step 4: Find and validate generated diagram
                diagram_path = self._find_diagram_file_enhanced(execution_result.get("expected_filename"))
                
                if not diagram_path:
                    if attempt < max_retries:
                        self.logger.warning(f"Diagram file not found on attempt {attempt + 1}, retrying")
                        await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
                        continue
                    
                    return DiagramResponse(
                        success=False,
                        error="Diagram file not found after execution",
                        execution_log=execution_result.get("log"),
                        validation_details=validation_result,
                        retry_count=attempt,
                        processing_time=time.time() - start_time
                    )
                
                # Step 5: Validate generated diagram file
                file_validation = self._validate_diagram_file(diagram_path)
                if not file_validation["valid"]:
                    if attempt < max_retries:
                        self.logger.warning(f"Invalid diagram file on attempt {attempt + 1}: {file_validation['error']}")
                        await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
                        continue
                    
                    return DiagramResponse(
                        success=False,
                        error=f"Invalid diagram file: {file_validation['error']}",
                        execution_log=execution_result.get("log"),
                        validation_details=validation_result,
                        retry_count=attempt,
                        processing_time=time.time() - start_time
                    )
                
                # Success!
                return DiagramResponse(
                    success=True,
                    diagram_path=str(diagram_path),
                    execution_log=execution_result.get("log"),
                    validation_details=validation_result,
                    retry_count=attempt,
                    processing_time=time.time() - start_time
                )
                
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
                    continue
                else:
                    return DiagramResponse(
                        success=False,
                        error=f"Unexpected error after {max_retries + 1} attempts: {str(e)}",
                        execution_log=traceback.format_exc(),
                        retry_count=attempt,
                        processing_time=time.time() - start_time
                    )
        
        # This should never be reached
        return DiagramResponse(
            success=False,
            error="Maximum retry attempts exceeded",
            retry_count=max_retries,
            processing_time=time.time() - start_time
        )    
    def _enhanced_validate_code(self, code: str) -> Dict[str, Any]:
        """
        Enhanced validation with detailed error reporting and fix suggestions.
        
        Args:
            code: Python code to validate
            
        Returns:
            Dictionary with detailed validation result
        """
        try:
            # Check for required imports
            if "from diagrams" not in code and "import diagrams" not in code:
                return {
                    "valid": False, 
                    "error": "Missing diagrams library import",
                    "retryable": True,
                    "suggestion": "Add 'from diagrams import Diagram'"
                }
            
            # Check for Diagram context manager
            if "with Diagram" not in code:
                return {
                    "valid": False, 
                    "error": "Missing Diagram context manager",
                    "retryable": True,
                    "suggestion": "Use 'with Diagram(...)' context manager"
                }
            
            # Check for show=False (required for web interface)
            if "show=False" not in code and "show = False" not in code:
                return {
                    "valid": False, 
                    "error": "Missing show=False parameter",
                    "retryable": True,
                    "suggestion": "Add show=False to Diagram constructor"
                }
            
            # Check for filename parameter
            if "filename=" not in code:
                return {
                    "valid": False, 
                    "error": "Missing filename parameter",
                    "retryable": True,
                    "suggestion": "Add filename='output/diagram_name' to Diagram constructor"
                }
            
            # Security checks - prevent dangerous operations
            dangerous_patterns = [
                ("import os", "Operating system access"),
                ("import subprocess", "Subprocess execution"),
                ("import sys", "System access"),
                ("exec(", "Code execution"),
                ("eval(", "Code evaluation"),
                ("__import__", "Dynamic imports"),
                ("open(", "File operations"),
                ("file(", "File operations"),
                ("input(", "User input"),
                ("raw_input(", "User input")
            ]
            
            for pattern, description in dangerous_patterns:
                if pattern in code:
                    return {
                        "valid": False, 
                        "error": f"Security risk: {description} ({pattern})",
                        "retryable": False,
                        "suggestion": f"Remove {pattern} from code"
                    }
            
            # Check for proper provider imports
            provider_patterns = ['.aws.', '.azure.', '.gcp.', '.k8s.', '.onprem.', '.oci.']
            if not any(provider in code for provider in provider_patterns):
                return {
                    "valid": False, 
                    "error": "No valid cloud provider components found",
                    "retryable": True,
                    "suggestion": "Import components from diagrams.aws, diagrams.azure, etc."
                }
            
            # Try to compile the code
            compile(code, "<string>", "exec")
            
            return {
                "valid": True,
                "checks_passed": [
                    "Diagrams import found",
                    "Diagram context manager found", 
                    "show=False parameter found",
                    "filename parameter found",
                    "No security risks detected",
                    "Valid provider components found",
                    "Syntax is valid"
                ]
            }
            
        except SyntaxError as e:
            return {
                "valid": False, 
                "error": f"Syntax error: {str(e)}",
                "retryable": True,
                "suggestion": "Fix Python syntax errors"
            }
        except Exception as e:
            return {
                "valid": False, 
                "error": f"Validation error: {str(e)}",
                "retryable": False
            }
    
    def _auto_fix_code(self, code: str, error: str) -> str:
        """
        Attempt to automatically fix common code issues.
        
        Args:
            code: Original code
            error: Error description
            
        Returns:
            Fixed code or original code if no fix possible
        """
        fixed_code = code
        
        # Fix missing show=False
        if "show=False" not in code and "with Diagram(" in code:
            fixed_code = fixed_code.replace(
                'with Diagram("',
                'with Diagram("'
            ).replace(
                'with Diagram(',
                'with Diagram('
            )
            # Add show=False if not present
            if "show=" not in fixed_code:
                fixed_code = fixed_code.replace(
                    'with Diagram("',
                    'with Diagram("'
                ).replace(
                    '):', 
                    ', show=False):'
                ).replace(
                    ', show=False, show=False)', 
                    ', show=False)'
                )
        
        # Fix missing filename
        if "filename=" not in code and "with Diagram(" in code:
            if ", show=False):" in fixed_code:
                fixed_code = fixed_code.replace(
                    ", show=False):",
                    ', show=False, filename="output/diagram"):'
                )
            elif "):" in fixed_code:
                fixed_code = fixed_code.replace(
                    "):",
                    ', filename="output/diagram"):'
                )
        
        return fixed_code
    
    def _prepare_code_enhanced(self, code: str, filename: Optional[str] = None) -> str:
        """
        Enhanced code preparation with better error handling and path management.
        
        Args:
            code: Original Python code
            filename: Optional custom filename
            
        Returns:
            Prepared code with proper imports and configuration
        """
        # Ensure output directory exists and clean imports
        prepared_code = f"""
import sys
import os
from pathlib import Path

# Ensure output directory exists
output_dir = Path('{self.output_dir}')
output_dir.mkdir(exist_ok=True)

# Set working directory for consistent behavior
os.chdir(output_dir.parent)

try:
{self._indent_code(code, '    ')}
except Exception as e:
    print(f"Error executing diagram code: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Diagram generation completed successfully")
"""
        
        # If filename is specified, try to replace it in the code
        if filename:
            base_filename = filename.replace('.png', '').replace('.jpg', '').replace('.svg', '').replace('.pdf', '')
            # Replace various filename patterns
            patterns = [
                (r'filename\s*=\s*["\'][^"\']*["\']', f'filename="output/{base_filename}"'),
                (r'filename\s*=\s*"output/[^"]*"', f'filename="output/{base_filename}"'),
                (r'filename\s*=\s*\'output/[^\']*\'', f'filename="output/{base_filename}"')
            ]
            
            import re
            for pattern, replacement in patterns:
                prepared_code = re.sub(pattern, replacement, prepared_code)
        
        return prepared_code
    
    def _indent_code(self, code: str, indent: str) -> str:
        """
        Indent all lines of code by the specified amount.
        
        Args:
            code: Code to indent
            indent: Indentation string
            
        Returns:
            Indented code
        """
        lines = code.split('\n')
        indented_lines = []
        
        for line in lines:
            if line.strip():  # Only indent non-empty lines
                indented_lines.append(indent + line)
            else:
                indented_lines.append(line)
        
        return '\n'.join(indented_lines)
    
    async def _execute_code_enhanced(self, code: str, attempt: int) -> Dict[str, Any]:
        """
        Enhanced code execution with better monitoring and error reporting.
        
        Args:
            code: Python code to execute
            attempt: Current attempt number
            
        Returns:
            Dictionary with detailed execution result
        """
        temp_file_path = None
        
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            self.logger.info(f"Executing diagram code (attempt {attempt + 1})")
            
            # Execute with enhanced monitoring
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=60,  # Increased timeout
                cwd=str(self.output_dir.parent),
                env={**os.environ, 'PYTHONPATH': str(Path(__file__).parent.parent)}
            )
            
            # Detailed result analysis
            if result.returncode != 0:
                error_analysis = self._analyze_execution_error(result.stderr, result.stdout)
                return {
                    "success": False,
                    "error": error_analysis["error"],
                    "log": result.stdout + result.stderr,
                    "retryable": error_analysis["retryable"],
                    "error_type": error_analysis["type"]
                }
            
            return {
                "success": True,
                "log": result.stdout,
                "expected_filename": self._extract_filename_from_code(code),
                "execution_time": "Success"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Code execution timed out (60 second limit)",
                "log": "Execution exceeded time limit - possible infinite loop or heavy computation",
                "retryable": False,
                "error_type": "timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution setup error: {str(e)}",
                "log": traceback.format_exc(),
                "retryable": True,
                "error_type": "setup"
            }
        finally:
            # Clean up temporary file
            if temp_file_path:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
    
    def _analyze_execution_error(self, stderr: str, stdout: str) -> Dict[str, Any]:
        """
        Analyze execution errors to determine if they're retryable and provide better messages.
        
        Args:
            stderr: Standard error output
            stdout: Standard output
            
        Returns:
            Dictionary with error analysis
        """
        combined_output = stderr + stdout
        
        # Import errors - usually retryable with different approach
        if "ImportError" in combined_output or "ModuleNotFoundError" in combined_output:
            if "diagrams" in combined_output:
                return {
                    "error": "Diagrams library not found or not properly installed",
                    "retryable": False,
                    "type": "import",
                    "suggestion": "Install diagrams library: pip install diagrams"
                }
            else:
                return {
                    "error": f"Missing Python module: {self._extract_missing_module(combined_output)}",
                    "retryable": True,
                    "type": "import"
                }
        
        # Graphviz errors
        if "graphviz" in combined_output.lower() or "dot" in combined_output:
            return {
                "error": "Graphviz not found - required for diagram rendering",
                "retryable": False,
                "type": "graphviz",
                "suggestion": "Install Graphviz: https://graphviz.gitlab.io/download/"
            }
        
        # Syntax errors in generated code
        if "SyntaxError" in combined_output:
            return {
                "error": f"Syntax error in generated code: {self._extract_syntax_error(combined_output)}",
                "retryable": True,
                "type": "syntax"
            }
        
        # Permission/file system errors
        if "PermissionError" in combined_output or "FileNotFoundError" in combined_output:
            return {
                "error": "File system access error - check permissions",
                "retryable": True,
                "type": "filesystem"
            }
        
        # Memory errors
        if "MemoryError" in combined_output:
            return {
                "error": "Out of memory - diagram too complex",
                "retryable": False,
                "type": "memory"
            }
        
        # Generic error
        return {
            "error": f"Execution failed: {stderr.strip() if stderr.strip() else stdout.strip()}",
            "retryable": True,
            "type": "unknown"
        }
    
    def _extract_missing_module(self, output: str) -> str:
        """Extract missing module name from error output."""
        import re
        match = re.search(r"No module named '([^']+)'", output)
        return match.group(1) if match else "unknown module"
    
    def _extract_syntax_error(self, output: str) -> str:
        """Extract syntax error details from output."""
        lines = output.split('\n')
        for line in lines:
            if "SyntaxError:" in line:
                return line.split("SyntaxError:")[-1].strip()
        return "Unknown syntax error"
    
    def _extract_filename_from_code(self, code: str) -> Optional[str]:
        """
        Extract the expected filename from the diagram code with better pattern matching.
        
        Args:
            code: Python code
            
        Returns:
            Expected filename or None
        """
        import re
        
        # Enhanced patterns for filename extraction
        patterns = [
            r'filename\s*=\s*["\']([^"\']+)["\']',
            r'filename\s*=\s*f["\']([^"\']+)["\']',
            r'Diagram\([^,]+,\s*[^,]*filename\s*=\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, code)
            if match:
                return match.group(1)
        
        return None
    
    def _find_diagram_file_enhanced(self, expected_filename: Optional[str] = None) -> Optional[Path]:
        """
        Enhanced diagram file discovery with multiple search strategies.
        
        Args:
            expected_filename: Expected filename from code
            
        Returns:
            Path to the diagram file or None
        """
        extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf', '.dot']
        search_paths = [self.output_dir, Path.cwd()]
        
        # Strategy 1: Look for expected filename
        if expected_filename:
            for search_path in search_paths:
                for ext in extensions:
                    # Direct match
                    file_path = search_path / (expected_filename + ext)
                    if file_path.exists():
                        return file_path
                    
                    # Match without path prefix
                    base_name = Path(expected_filename).name
                    file_path = search_path / (base_name + ext)
                    if file_path.exists():
                        return file_path
        
        # Strategy 2: Look for most recent diagram file
        all_files = []
        for search_path in search_paths:
            for ext in extensions:
                all_files.extend(search_path.glob(f'*{ext}'))
        
        if all_files:
            # Filter out system files and return most recent
            diagram_files = [f for f in all_files if not f.name.startswith('.')]
            if diagram_files:
                return max(diagram_files, key=lambda p: p.stat().st_mtime)
        
        return None
    
    def _validate_diagram_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate that the generated diagram file is valid.
        
        Args:
            file_path: Path to the diagram file
            
        Returns:
            Validation result dictionary
        """
        try:
            if not file_path.exists():
                return {"valid": False, "error": "File does not exist"}
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                return {"valid": False, "error": "Generated file is empty"}
            
            if file_size < 100:  # Very small files are likely errors
                return {"valid": False, "error": "Generated file is too small (likely an error)"}
            
            # Check file extension
            valid_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf', '.dot']
            if file_path.suffix.lower() not in valid_extensions:
                return {"valid": False, "error": f"Invalid file extension: {file_path.suffix}"}
            
            # Basic content validation for image files
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                with open(file_path, 'rb') as f:
                    header = f.read(8)
                    if file_path.suffix.lower() == '.png' and not header.startswith(b'\x89PNG'):
                        return {"valid": False, "error": "Invalid PNG file header"}
                    elif file_path.suffix.lower() in ['.jpg', '.jpeg'] and not header.startswith(b'\xff\xd8'):
                        return {"valid": False, "error": "Invalid JPEG file header"}
            
            return {
                "valid": True,
                "file_size": file_size,
                "file_type": file_path.suffix,
                "file_path": str(file_path)
            }
            
        except Exception as e:
            return {"valid": False, "error": f"File validation error: {str(e)}"}
    
    def list_diagrams(self) -> List[Dict[str, Any]]:
        """
        List all generated diagrams with enhanced metadata.
        
        Returns:
            List of diagram information dictionaries
        """
        diagrams = []
        extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf', '.dot']
        
        try:
            for ext in extensions:
                for file_path in self.output_dir.glob(f'*{ext}'):
                    try:
                        stat = file_path.stat()
                        diagrams.append({
                            "filename": file_path.name,
                            "path": str(file_path),
                            "size": stat.st_size,
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "created": stat.st_mtime,
                            "modified": stat.st_mtime,
                            "extension": ext,
                            "valid": self._validate_diagram_file(file_path)["valid"]
                        })
                    except Exception as e:
                        self.logger.warning(f"Error getting info for {file_path}: {e}")
            
            # Sort by creation time (newest first)
            diagrams.sort(key=lambda x: x["created"], reverse=True)
            return diagrams
            
        except Exception as e:
            self.logger.error(f"Error listing diagrams: {str(e)}")
            return []
    
    def cleanup_old_diagrams(self, max_files: int = 50) -> int:
        """
        Clean up old diagram files to prevent disk space issues.
        
        Args:
            max_files: Maximum number of files to keep
            
        Returns:
            Number of files cleaned up
        """
        try:
            diagrams = self.list_diagrams()
            if len(diagrams) <= max_files:
                return 0
            
            # Remove oldest files
            files_to_remove = diagrams[max_files:]
            removed_count = 0
            
            for file_info in files_to_remove:
                try:
                    Path(file_info["path"]).unlink()
                    removed_count += 1
                    self.logger.info(f"Removed old diagram: {file_info['filename']}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file_info['filename']}: {e}")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up diagrams: {str(e)}")
            return 0
