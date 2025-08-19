"""
OpenSCENARIO 2.0 Parser Integration and Validation System
Handles py-osc2 parser integration with feedback loop for error correction.
"""

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import re

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result from OpenSCENARIO validation."""
    is_valid: bool
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    line_number: Optional[int] = None
    suggestions: List[str] = None
    validation_time: float = 0.0

class OSCValidator:
    """OpenSCENARIO 2.0 code validator using py-osc2."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.parser_command = config.get('validation.parser_command', 'py-osc2')
        self.timeout = config.get('validation.timeout_seconds', 30)
        self.max_retries = config.get('validation.max_retries', 3)
        
        # Common error patterns and fixes
        self.error_patterns = self._initialize_error_patterns()
        
        logger.info("OSCValidator initialized")
    
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common error patterns and their fixes."""
        return {
            'syntax_error': {
                'patterns': [
                    r'syntax error.*line (\d+)',
                    r'unexpected token.*line (\d+)',
                    r'parse error.*line (\d+)'
                ],
                'suggestions': [
                    "Check for missing semicolons or brackets",
                    "Verify proper OpenSCENARIO 2.0 syntax",
                    "Ensure all blocks are properly closed"
                ]
            },
            'undefined_entity': {
                'patterns': [
                    r'undefined.*entity.*(\w+)',
                    r'unknown.*reference.*(\w+)',
                    r'cannot resolve.*(\w+)'
                ],
                'suggestions': [
                    "Define the entity before using it",
                    "Check entity name spelling",
                    "Ensure proper import statements"
                ]
            },
            'type_mismatch': {
                'patterns': [
                    r'type mismatch.*expected.*(\w+).*got.*(\w+)',
                    r'incompatible types.*(\w+).*(\w+)'
                ],
                'suggestions': [
                    "Check parameter types match expected values",
                    "Convert values to correct type",
                    "Review OpenSCENARIO 2.0 type system"
                ]
            },
            'missing_required': {
                'patterns': [
                    r'missing required.*(\w+)',
                    r'required.*(\w+).*not specified'
                ],
                'suggestions': [
                    "Add the required parameter or attribute",
                    "Check OpenSCENARIO 2.0 specification for required fields"
                ]
            }
        }
    
    def validate_code(self, osc_code: str, filename: str = None) -> ValidationResult:
        """Validate OpenSCENARIO 2.0 code using py-osc2.
        
        Args:
            osc_code: OpenSCENARIO code to validate
            filename: Optional filename for the code
            
        Returns:
            Validation result with error details if any
        """
        start_time = time.time()
        
        if not osc_code.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Empty code provided",
                validation_time=time.time() - start_time
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.osc', delete=False) as f:
            f.write(osc_code)
            temp_file = f.name
        
        try:
            result = self._run_parser(temp_file)
            result.validation_time = time.time() - start_time
            return result
            
        finally:
            # Clean up temporary file
            Path(temp_file).unlink(missing_ok=True)
    
    def _run_parser(self, file_path: str) -> ValidationResult:
        """Run py-osc2 parser on file."""
        try:
            # Run parser command
            cmd = [self.parser_command, file_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                # Parsing successful
                return ValidationResult(is_valid=True)
            else:
                # Parse error occurred
                error_output = result.stderr or result.stdout
                return self._parse_error_output(error_output)
                
        except subprocess.TimeoutExpired:
            return ValidationResult(
                is_valid=False,
                error_message="Parser timeout - code may be too complex",
                error_type="timeout"
            )
        except FileNotFoundError:
            return ValidationResult(
                is_valid=False,
                error_message=f"Parser command not found: {self.parser_command}",
                error_type="parser_not_found"
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Parser execution failed: {str(e)}",
                error_type="execution_error"
            )
    
    def _parse_error_output(self, error_output: str) -> ValidationResult:
        """Parse error output from py-osc2 to extract useful information."""
        if not error_output:
            return ValidationResult(
                is_valid=False,
                error_message="Unknown parser error"
            )
        
        # Extract line number if present
        line_number = None
        line_match = re.search(r'line (\d+)', error_output)
        if line_match:
            line_number = int(line_match.group(1))
        
        # Categorize error and get suggestions
        error_type, suggestions = self._categorize_error(error_output)
        
        return ValidationResult(
            is_valid=False,
            error_message=error_output.strip(),
            error_type=error_type,
            line_number=line_number,
            suggestions=suggestions or []
        )
    
    def _categorize_error(self, error_output: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """Categorize error and provide suggestions."""
        error_lower = error_output.lower()
        
        for error_type, config in self.error_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, error_lower):
                    return error_type, config['suggestions']
        
        return None, None
    
    def validate_with_retry(self, osc_code: str, max_attempts: int = None) -> Tuple[ValidationResult, List[str]]:
        """Validate code with retry attempts for common fixes.
        
        Args:
            osc_code: OpenSCENARIO code to validate
            max_attempts: Maximum retry attempts (uses config default if None)
            
        Returns:
            Tuple of (final_validation_result, list_of_attempted_fixes)
        """
        if max_attempts is None:
            max_attempts = self.max_retries
        
        attempted_fixes = []
        current_code = osc_code
        
        for attempt in range(max_attempts):
            result = self.validate_code(current_code)
            
            if result.is_valid:
                return result, attempted_fixes
            
            # Try to apply automatic fixes
            if attempt < max_attempts - 1:
                fixed_code, fix_description = self._attempt_auto_fix(current_code, result)
                
                if fixed_code != current_code:
                    attempted_fixes.append(fix_description)
                    current_code = fixed_code
                    logger.debug(f"Applied fix: {fix_description}")
                else:
                    # No fix could be applied, break early
                    break
        
        return result, attempted_fixes
    
    def _attempt_auto_fix(self, code: str, validation_result: ValidationResult) -> Tuple[str, str]:
        """Attempt to automatically fix common errors.
        
        Args:
            code: Current code
            validation_result: Validation result with error info
            
        Returns:
            Tuple of (potentially_fixed_code, fix_description)
        """
        if not validation_result.error_type:
            return code, "No automatic fix available"
        
        # Try fixes based on error type
        if validation_result.error_type == 'syntax_error':
            return self._fix_syntax_errors(code, validation_result)
        elif validation_result.error_type == 'missing_required':
            return self._fix_missing_required(code, validation_result)
        elif validation_result.error_type == 'undefined_entity':
            return self._fix_undefined_entity(code, validation_result)
        
        return code, f"No fix available for error type: {validation_result.error_type}"
    
    def _fix_syntax_errors(self, code: str, result: ValidationResult) -> Tuple[str, str]:
        """Attempt to fix common syntax errors."""
        lines = code.split('\n')
        
        # Common fixes
        fixes_applied = []
        
        # Fix missing semicolons
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped and 
                not stripped.endswith((';', '{', '}', ':', '//')) and 
                not stripped.startswith(('import', 'using', '//', '#'))):
                lines[i] = line + ';'
                fixes_applied.append(f"Added semicolon to line {i+1}")
        
        # Fix common bracket issues
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces > close_braces:
            lines.append('}')
            fixes_applied.append("Added missing closing brace")
        elif close_braces > open_braces:
            lines.insert(0, '{')
            fixes_applied.append("Added missing opening brace")
        
        fixed_code = '\n'.join(lines)
        fix_description = '; '.join(fixes_applied) if fixes_applied else "No syntax fixes applied"
        
        return fixed_code, fix_description
    
    def _fix_missing_required(self, code: str, result: ValidationResult) -> Tuple[str, str]:
        """Attempt to fix missing required parameters."""
        # Extract missing parameter from error message
        if result.error_message:
            match = re.search(r'missing required.*(\w+)', result.error_message.lower())
            if match:
                missing_param = match.group(1)
                
                # Add common default values for known parameters
                defaults = {
                    'name': '"DefaultName"',
                    'id': '"default_id"',
                    'type': '"DefaultType"',
                    'value': '0.0',
                    'duration': '1.0',
                    'speed': '10.0'
                }
                
                if missing_param in defaults:
                    # Simple insertion - this could be more sophisticated
                    default_value = defaults[missing_param]
                    fixed_code = code + f'\n    {missing_param}: {default_value};'
                    return fixed_code, f"Added default value for {missing_param}"
        
        return code, "Could not fix missing required parameter"
    
    def _fix_undefined_entity(self, code: str, result: ValidationResult) -> Tuple[str, str]:
        """Attempt to fix undefined entity references."""
        if result.error_message:
            match = re.search(r'undefined.*entity.*(\w+)', result.error_message.lower())
            if match:
                entity_name = match.group(1)
                
                # Add a basic entity definition
                entity_def = f"""
entity {entity_name} {{
    // Auto-generated entity definition
    name: "{entity_name}";
}};
"""
                fixed_code = entity_def + '\n' + code
                return fixed_code, f"Added basic definition for entity {entity_name}"
        
        return code, "Could not fix undefined entity"
    
    def generate_error_feedback(self, validation_result: ValidationResult, 
                              original_query: str) -> str:
        """Generate feedback message for LLM based on validation errors.
        
        Args:
            validation_result: Validation result with errors
            original_query: Original user query
            
        Returns:
            Formatted feedback message for LLM
        """
        if validation_result.is_valid:
            return ""
        
        feedback_parts = [
            "The generated OpenSCENARIO 2.0 code has validation errors:",
            f"Error: {validation_result.error_message}"
        ]
        
        if validation_result.line_number:
            feedback_parts.append(f"Line number: {validation_result.line_number}")
        
        if validation_result.suggestions:
            feedback_parts.append("Suggestions:")
            for suggestion in validation_result.suggestions:
                feedback_parts.append(f"- {suggestion}")
        
        feedback_parts.extend([
            "",
            "Please correct the code according to OpenSCENARIO 2.0 specification.",
            "Focus on:",
            "- Proper syntax and structure",
            "- Required parameters and attributes", 
            "- Valid entity references",
            "- Correct data types",
            "",
            f"Original request: {original_query}"
        ])
        
        return "\n".join(feedback_parts)
    
    def check_parser_availability(self) -> bool:
        """Check if py-osc2 parser is available."""
        try:
            result = subprocess.run(
                [self.parser_command, '--help'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_parser_version(self) -> Optional[str]:
        """Get py-osc2 parser version."""
        try:
            result = subprocess.run(
                [self.parser_command, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics."""
        return {
            'parser_command': self.parser_command,
            'parser_available': self.check_parser_availability(),
            'parser_version': self.get_parser_version(),
            'timeout_seconds': self.timeout,
            'max_retries': self.max_retries,
            'error_patterns_count': len(self.error_patterns)
        }
