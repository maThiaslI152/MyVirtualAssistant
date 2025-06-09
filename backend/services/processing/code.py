from typing import List, Dict, Any, Optional
import logging
import ast
import re
from dataclasses import dataclass
import tree_sitter
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class CodeBlock:
    code: str
    language: str
    start_line: int
    end_line: int
    metadata: Dict[str, Any] = None

class CodeProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize code understanding models
        self.code_analyzer = pipeline("code-analysis", model="microsoft/codebert-base")
        self.code_generator = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-py")
        self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")
        
        # Initialize tree-sitter for code parsing
        self.parser = tree_sitter.Parser()
        self.parser.set_language(tree_sitter.Language('build/my-languages.so', 'python'))
        
    def analyze_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Analyze code and extract various features."""
        try:
            # Parse code
            tree = self.parser.parse(bytes(code, 'utf8'))
            
            # Extract features
            features = {
                'complexity': self._analyze_complexity(tree),
                'structure': self._analyze_structure(tree),
                'dependencies': self._extract_dependencies(code, language),
                'metrics': self._calculate_metrics(tree),
                'suggestions': self._generate_suggestions(code, language)
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error analyzing code: {str(e)}")
            return {}
            
    def generate_code(self, prompt: str, language: str = "python") -> str:
        """Generate code based on natural language prompt."""
        try:
            # Prepare input
            inputs = self.code_tokenizer(prompt, return_tensors="pt")
            
            # Generate code
            outputs = self.code_generator.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
            
            # Decode and format
            generated_code = self.code_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._format_code(generated_code, language)
            
        except Exception as e:
            self.logger.error(f"Error generating code: {str(e)}")
            return ""
            
    def _analyze_complexity(self, tree: tree_sitter.Tree) -> Dict[str, Any]:
        """Analyze code complexity."""
        try:
            # Calculate cyclomatic complexity
            complexity = 0
            for node in tree.root_node.traverse():
                if node.type in ['if_statement', 'for_statement', 'while_statement', 'case_statement']:
                    complexity += 1
                    
            # Calculate nesting depth
            max_depth = 0
            current_depth = 0
            for node in tree.root_node.traverse():
                if node.type in ['function_definition', 'class_definition', 'if_statement', 'for_statement', 'while_statement']:
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif node.type == '}':
                    current_depth -= 1
                    
            return {
                'cyclomatic_complexity': complexity,
                'max_nesting_depth': max_depth,
                'complexity_score': complexity * 0.7 + max_depth * 0.3
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing complexity: {str(e)}")
            return {}
            
    def _analyze_structure(self, tree: tree_sitter.Tree) -> Dict[str, Any]:
        """Analyze code structure."""
        try:
            structure = {
                'functions': [],
                'classes': [],
                'imports': [],
                'variables': []
            }
            
            for node in tree.root_node.traverse():
                if node.type == 'function_definition':
                    structure['functions'].append({
                        'name': node.child_by_field_name('name').text.decode('utf8'),
                        'parameters': self._extract_parameters(node),
                        'return_type': self._extract_return_type(node)
                    })
                elif node.type == 'class_definition':
                    structure['classes'].append({
                        'name': node.child_by_field_name('name').text.decode('utf8'),
                        'methods': self._extract_methods(node)
                    })
                elif node.type == 'import_statement':
                    structure['imports'].append(node.text.decode('utf8'))
                elif node.type == 'variable_declaration':
                    structure['variables'].append({
                        'name': node.child_by_field_name('name').text.decode('utf8'),
                        'type': self._extract_variable_type(node)
                    })
                    
            return structure
            
        except Exception as e:
            self.logger.error(f"Error analyzing structure: {str(e)}")
            return {}
            
    def _extract_dependencies(self, code: str, language: str) -> List[str]:
        """Extract code dependencies."""
        try:
            if language == "python":
                # Parse imports
                tree = ast.parse(code)
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(n.name for n in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        imports.append(f"{node.module}.{node.names[0].name}")
                return imports
            else:
                # Add support for other languages
                return []
                
        except Exception as e:
            self.logger.error(f"Error extracting dependencies: {str(e)}")
            return []
            
    def _calculate_metrics(self, tree: tree_sitter.Tree) -> Dict[str, float]:
        """Calculate various code metrics."""
        try:
            metrics = {
                'lines_of_code': 0,
                'comment_ratio': 0,
                'function_count': 0,
                'class_count': 0,
                'average_function_length': 0,
                'average_class_length': 0
            }
            
            # Count lines and comments
            total_lines = 0
            comment_lines = 0
            function_lengths = []
            class_lengths = []
            
            for node in tree.root_node.traverse():
                if node.type == 'function_definition':
                    metrics['function_count'] += 1
                    function_lengths.append(node.end_point[0] - node.start_point[0])
                elif node.type == 'class_definition':
                    metrics['class_count'] += 1
                    class_lengths.append(node.end_point[0] - node.start_point[0])
                elif node.type == 'comment':
                    comment_lines += 1
                total_lines += 1
                
            # Calculate metrics
            metrics['lines_of_code'] = total_lines
            metrics['comment_ratio'] = comment_lines / total_lines if total_lines > 0 else 0
            metrics['average_function_length'] = np.mean(function_lengths) if function_lengths else 0
            metrics['average_class_length'] = np.mean(class_lengths) if class_lengths else 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}
            
    def _generate_suggestions(self, code: str, language: str) -> List[str]:
        """Generate code improvement suggestions."""
        try:
            suggestions = []
            
            # Analyze code style
            if language == "python":
                import pylint.lint
                from io import StringIO
                from contextlib import redirect_stdout
                
                # Run pylint
                with StringIO() as output:
                    with redirect_stdout(output):
                        pylint.lint.Run([code], do_exit=False)
                    suggestions.extend(output.getvalue().split('\n'))
                    
            # Add language-specific suggestions
            if language == "python":
                # Check for common Python issues
                if "import *" in code:
                    suggestions.append("Avoid using 'import *' as it can lead to namespace pollution")
                if "eval(" in code:
                    suggestions.append("Avoid using eval() as it can be a security risk")
                    
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {str(e)}")
            return []
            
    def _format_code(self, code: str, language: str) -> str:
        """Format generated code according to language standards."""
        try:
            if language == "python":
                import black
                return black.format_str(code, mode=black.FileMode())
            elif language == "javascript":
                import jsbeautifier
                return jsbeautifier.beautify(code)
            else:
                return code
                
        except Exception as e:
            self.logger.error(f"Error formatting code: {str(e)}")
            return code
            
    def _extract_parameters(self, node: tree_sitter.Node) -> List[Dict[str, str]]:
        """Extract function parameters."""
        try:
            parameters = []
            params_node = node.child_by_field_name('parameters')
            if params_node:
                for param in params_node.children:
                    if param.type == 'parameter':
                        param_name = param.child_by_field_name('name').text.decode('utf8')
                        param_type = param.child_by_field_name('type').text.decode('utf8') if param.child_by_field_name('type') else 'any'
                        parameters.append({
                            'name': param_name,
                            'type': param_type
                        })
            return parameters
        except Exception as e:
            self.logger.error(f"Error extracting parameters: {str(e)}")
            return []
            
    def _extract_return_type(self, node: tree_sitter.Node) -> str:
        """Extract function return type."""
        try:
            return_node = node.child_by_field_name('return_type')
            return return_node.text.decode('utf8') if return_node else 'any'
        except Exception as e:
            self.logger.error(f"Error extracting return type: {str(e)}")
            return 'any'
            
    def _extract_methods(self, node: tree_sitter.Node) -> List[Dict[str, Any]]:
        """Extract class methods."""
        try:
            methods = []
            body = node.child_by_field_name('body')
            if body:
                for child in body.children:
                    if child.type == 'function_definition':
                        methods.append({
                            'name': child.child_by_field_name('name').text.decode('utf8'),
                            'parameters': self._extract_parameters(child),
                            'return_type': self._extract_return_type(child)
                        })
            return methods
        except Exception as e:
            self.logger.error(f"Error extracting methods: {str(e)}")
            return []
            
    def _extract_variable_type(self, node: tree_sitter.Node) -> str:
        """Extract variable type."""
        try:
            type_node = node.child_by_field_name('type')
            return type_node.text.decode('utf8') if type_node else 'any'
        except Exception as e:
            self.logger.error(f"Error extracting variable type: {str(e)}")
            return 'any' 