#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EasyPeasyHindi v2.6
A small, playful command engine inspired by Sanskrit grammar.
Supports HYBRID & SHUDDH modes with real execution.
"""

import json
import sys
import re
from typing import Dict, Optional, Tuple, Any, List
from enum import Enum


class ExecutionMode(Enum):
    """Execution modes for the combinator."""
    HYBRID = "hybrid"   # Standard symbols (=, +, -, etc.)
    SHUDDH = "shuddh"   # Pure Sanskrit words (sama, yuj, viyuj, etc.)


class OperatorNormalizer:
    """Normalizes Sanskrit operators to standard mathematical symbols."""
    
    ASSIGNMENT_MAP = {
        'sama': '=',      # Assignment
    }
    
    COMPARISON_MAP = {
        'tulya': '==',    # Equality
        'adhika': '>',    # Greater than
        'nyuna': '<',     # Less than
        'adhika-tulya': '>=',  # Greater than or equal
        'nyuna-tulya': '<=',   # Less than or equal
        'na-tulya': '!=',      # Not equal
    }
    
    ARITHMETIC_MAP = {
        'yuj': '+',       # Addition
        'viyuj': '-',     # Subtraction
        'gun': '*',       # Multiplication
        'bhaj': '/',      # Division
        'shesh': '%',     # Modulo
        'ghat': '**',     # Exponentiation
    }
    
    # Combined map for all operators
    ALL_OPERATORS = {
        **ASSIGNMENT_MAP,
        **COMPARISON_MAP,
        **ARITHMETIC_MAP
    }
    
    # Reverse map (symbol to Sanskrit)
    REVERSE_MAP = {v: k for k, v in ALL_OPERATORS.items()}
    
    @staticmethod
    def normalize_operator(operator: str, mode: ExecutionMode) -> str:
        """
        Ye function mode ke hisaab se operators ko normalize karta hai:
        SHUDDH Mode mein: Sanskrit shabd (jaise 'sama') ko standard symbol (jaise '=') mein badal deta hai.
        HYBRID Mode mein: Jo bhi input hai use vaise ka vaisa (as-is) rehne deta hai.
        Arguments:
        operator: Input operator (example: 'sama', '=', 'yuj', '+').
        mode: Abhi ka execution mode.
        Returns:
        Normalized operator (jo hamesha ek standard symbol hi hoga).
        """
        if mode == ExecutionMode.SHUDDH:
            # In SHUDDH mode, convert Sanskrit to standard
            return OperatorNormalizer.ALL_OPERATORS.get(operator, operator)
        else:
            # In HYBRID mode, return as-is
            return operator
    
    @staticmethod
    def is_valid_operator(operator: str, mode: ExecutionMode) -> bool:
        """will check if operator is valid for the current mode."""
        if mode == ExecutionMode.SHUDDH:
            # Must be Sanskrit operator
            return operator in OperatorNormalizer.ALL_OPERATORS
        else:
            # Can be standard symbol or Sanskrit
            return operator in OperatorNormalizer.REVERSE_MAP or operator in OperatorNormalizer.ALL_OPERATORS
    
    @staticmethod
    def get_operator_info() -> str:
        """Get human-readable operator mapping information."""
        info = []
        info.append("Sanskrit Operator Mappings (SHUDDH Mode):")
        
        info.append("\nAssignment:")
        for sanskrit, symbol in OperatorNormalizer.ASSIGNMENT_MAP.items():
            info.append(f"  {sanskrit:15} → {symbol:3}")
        
        info.append("\nComparison:")
        for sanskrit, symbol in OperatorNormalizer.COMPARISON_MAP.items():
            info.append(f"  {sanskrit:15} → {symbol:3}")
        
        info.append("\nArithmetic:")
        for sanskrit, symbol in OperatorNormalizer.ARITHMETIC_MAP.items():
            info.append(f"  {sanskrit:15} → {symbol:3}")
        
        return '\n'.join(info)


class StatementValidator:
    """Validates stats based on execution mode."""
    
    @staticmethod
    def validate_statement(input_string: str, mode: ExecutionMode) -> Tuple[str, bool]:
"""
Statement ko validate karega aur asli command extract karega.

SHUDDH mode mein: Aakhri mein Purna Virama (|) hona chahiye.
HYBRID mode mein: Kisi terminator ki zaroorat nahi hai.

Returns:
    (cleaned_statement, is_script_end) ka ek Tuple.
"""
        # Deergha Virama (||) - script end signal
        if input_string.strip().endswith('||'):
            cleaned = input_string.strip()[:-2].strip()
            return cleaned, True
        
        # Purna Virama (|)
        if input_string.strip().endswith('|'):
            cleaned = input_string.strip()[:-1].strip()
            return cleaned, False
        
        # No terminator found
        if mode == ExecutionMode.SHUDDH:
            raise ValueError("SHUDDH mode requires Purna Virama (|) at statement end")
        
        return input_string.strip(), False
    
    @staticmethod
    def add_terminator(statement: str, mode: ExecutionMode, is_script_end: bool = False) -> str:
        """Add appropriate terminator to statement."""
        if mode == ExecutionMode.SHUDDH:
            if is_script_end:
                return f"{statement} ||"
            else:
                return f"{statement} |"
        return statement


class ASCIINormalizer:
    """Converts QWERTY-friendly input to Sanskrit-standard transliterations."""
    
    # Mapping from ASCII-friendly
    DHATU_MAP = {
        'vad': 'vad',     
        'kr': 'kṛ',        # ṛ vowel
        'stha': 'sthā',    # ā vowel
        'bhu': 'bhū',      # ū vowel
        'path': 'paṭh',    # ṭ consonant
        'gan': 'gaṇ',      # ṇ consonant
        'drs': 'dṛś',      # ṛ vowel + ś
        'drsh': 'dṛś',     # Alternative spelling
        'cal': 'cal',    
        'man': 'man',     
    }
    
    UPASARGA_MAP = {
        'pra': 'pra',
        'anu': 'anu',
        'prati': 'prati',
        'vi': 'vi',
        'sam': 'sam',
        'ava': 'ava',
    }
    
    PRATYAYA_MAP = {
        'atu': 'atu',
        'ati': 'ati',
        'ane': 'aṇe',      # ṇ consonant
        'et': 'et',
    }
    
    @staticmethod
    def normalize_command(command: str) -> str:
"""
ASCII-friendly command ko proper Sanskrit transliteration mein badlega.

Examples:
    "bhu-ati" -> "bhū-ati"
    "kr-atu" -> "kṛ-atu"
    "pra-gan-atu" -> "pra-gaṇ-atu"
    "path-ane" -> "paṭh-aṇe"

Args:
    command: ASCII format mein input command.

Returns:
    Proper Sanskrit transliteration ke saath normalized command.
"""
        parts = command.strip().split('-')
        normalized_parts = []
        
        if len(parts) == 3:
            # prefix-dhatu-suffix
            prefix = ASCIINormalizer.UPASARGA_MAP.get(parts[0], parts[0])
            dhatu = ASCIINormalizer.DHATU_MAP.get(parts[1], parts[1])
            suffix = ASCIINormalizer.PRATYAYA_MAP.get(parts[2], parts[2])
            normalized_parts = [prefix, dhatu, suffix]
        elif len(parts) == 2:
            # dhatu-suffix
            dhatu = ASCIINormalizer.DHATU_MAP.get(parts[0], parts[0])
            suffix = ASCIINormalizer.PRATYAYA_MAP.get(parts[1], parts[1])
            normalized_parts = [dhatu, suffix]
        else:
            return command
        
        return '-'.join(normalized_parts)
    
    @staticmethod
    def get_mapping_info() -> str:
        """Get human-readable mapping information."""
        info = []
        info.append("ASCII → Sanskrit Mappings:")
        info.append("\nDhatus (Verbs):")
        for ascii_form, sanskrit in ASCIINormalizer.DHATU_MAP.items():
            if ascii_form != sanskrit:
                info.append(f"  {ascii_form:8} → {sanskrit}")
        
        info.append("\nPratyayas (Suffixes):")
        for ascii_form, sanskrit in ASCIINormalizer.PRATYAYA_MAP.items():
            if ascii_form != sanskrit:
                info.append(f"  {ascii_form:8} → {sanskrit}")
        
        return '\n'.join(info)


class GlobalMemory:
    """Global store mem. for var mangement."""
    
    def __init__(self):
        self.variables = {}
        self.history = []
    
    def set(self, name: str, value: Any):
        """Store a var in mem"""
        self.variables[name] = value
        self.history.append(('SET', name, value))
    
    def get(self, name: str) -> Any:
        """Retrieve a var from mem"""
        if name not in self.variables:
            raise KeyError(f"Variable '{name}' not found in memory")
        return self.variables[name]
    
    def exists(self, name: str) -> bool:
        """it will check if var exists"""
        return name in self.variables
    
    def delete(self, name: str):
        """Delete a var"""
        if name in self.variables:
            del self.variables[name]
            self.history.append(('DELETE', name, None))
    
    def clear(self):
        """Clear all vars"""
        self.variables.clear()
        self.history.clear()
    
    def list_all(self) -> Dict:
        """List all vars"""
        return self.variables.copy()
    
    def get_history(self) -> List:
        """Get operation history"""
        return self.history.copy()


class ArgumentParser:
    """Parse command arguments and data."""
    
    @staticmethod
    def extract_arguments(input_string: str) -> Tuple[str, List[str]]:
"""
        Input string mein se command aur arguments ko alag-alag nikaalein.
        
        Examples:
            "anu-vad-atu Hello" -> ("anu-vad-atu", ["Hello"])
            "bhū-ati x = 10" -> ("bhū-ati", ["x", "=", "10"])
            "vad-atu x" -> ("vad-atu", ["x"])
            'vad-atu "Hello World"' -> ("vad-atu", ["Hello World"])
        
        Returns:
            (command, arguments_list) ka ek Tuple.
"""
        input_string = input_string.strip()
        
        # Match quoted strings and non-space sequences
        tokens = re.findall(r'"[^"]*"|\S+', input_string)
        
        if not tokens:
            raise ValueError("Empty input")
        
        command = tokens[0]
        
        arguments = []
        for token in tokens[1:]:
            if token.startswith('"') and token.endswith('"'):
                arguments.append(token[1:-1])
            else:
                arguments.append(token)
        
        return command, arguments
    
    @staticmethod
    def parse_assignment(arguments: List[str], mode: ExecutionMode) -> Tuple[str, Any]:
"""
        Mode ke hisaab se assignment expressions ko parse karega.
        
        HYBRID: x = 10
        SHUDDH: x sama 10
        
        Returns:
            (var_name, value) ka ek Tuple.
"""
        if len(arguments) < 3:
            raise ValueError("Invalid assignment format. Expected: var operator value")
        
        var_name = arguments[0]
        operator = arguments[1]
        
        #normalized 
        normalized_op = OperatorNormalizer.normalize_operator(operator, mode)
        
        if normalized_op != '=':
            raise ValueError(f"Invalid assignment operator: '{operator}'. Expected '=' or 'sama'")
        
        #join meeting parts as values
        value_str = ' '.join(arguments[2:])
        
        # will try to parse as number
        try:
            if '.' in value_str:
                return var_name, float(value_str)
            else:
                return var_name, int(value_str)
        except ValueError:
            # Return as string (remove quotes if present)
            if value_str.startswith('"') and value_str.endswith('"'):
                return var_name, value_str[1:-1]
            return var_name, value_str


class ConditionalParser:
    """Parse and evaluate conditional expressions."""
    
    @staticmethod
    def parse_conditional(arguments: List[str]) -> Tuple[List[str], List[str]]:
        """
        Parse conditional expression into condition and action parts.
        
        Format: condition_parts action_command action_args
        Example: x > 5 vad-atu "Hello"
                 -> (["x", ">", "5"], ["vad-atu", "Hello"])
        
        Returns:
            Tuple of (condition_tokens, action_tokens)
        """
        # Find where the action command starts (look for command pattern with hyphens)
        action_start_idx = None
        for i, arg in enumerate(arguments):
            if '-' in arg and any(suffix in arg for suffix in ['atu', 'ati', 'ane', 'et', 'aṇe']):
                action_start_idx = i
                break
        
        if action_start_idx is None:
            raise ValueError("No action command found in conditional expression")
        
        condition_tokens = arguments[:action_start_idx]
        action_tokens = arguments[action_start_idx:]
        
        return condition_tokens, action_tokens
    
    @staticmethod
    def evaluate_condition(condition_tokens: List[str], memory: GlobalMemory, mode: ExecutionMode) -> bool:
        """

        """
        if len(condition_tokens) < 3:
            raise ValueError("Invalid condition format. Expected: operand1 operator operand2")
        
        # Get left operand
        left_str = condition_tokens[0]
        if memory.exists(left_str):
            left = memory.get(left_str)
        else:
            left = ConditionalParser._parse_value(left_str)
        
        # Get operator and normalize
        operator = condition_tokens[1]
        normalized_op = OperatorNormalizer.normalize_operator(operator, mode)
        
        # Get right operand
        right_str = condition_tokens[2]
        if memory.exists(right_str):
            right = memory.get(right_str)
        else:
            right = ConditionalParser._parse_value(right_str)
        
        # Evaluate
        if normalized_op == '==':
            return left == right
        elif normalized_op == '!=':
            return left != right
        elif normalized_op == '<':
            return left < right
        elif normalized_op == '>':
            return left > right
        elif normalized_op == '<=':
            return left <= right
        elif normalized_op == '>=':
            return left >= right
        else:
            raise ValueError(f"Unknown comparison operator: {operator} (normalized: {normalized_op})")
    
    @staticmethod
    def _parse_value(value_str: str) -> Any:
        """Parse string value to appropriate type."""
        # Try number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            # Return as string (remove quotes if present)
            if value_str.startswith('"') and value_str.endswith('"'):
                return value_str[1:-1]
            return value_str


class ExecutionEngine:
    """Execute resolved commands with actual logic."""
    
    def __init__(self, memory: GlobalMemory, mode: ExecutionMode = ExecutionMode.HYBRID):
        self.memory = memory
        self.output_buffer = []
        self.mode = mode
    
    def set_mode(self, mode: ExecutionMode):
        """Set the execution mode."""
        self.mode = mode
    
    def execute(self, components: Dict, arguments: List[str]) -> Dict:
        """
        Execute the command based on resolved components and arguments.
        
        Returns:
            Dictionary with execution results
        """
        dhatu_tag = components['dhatu']['tag']
        suffix_mode = components['suffix']['mode']
        prefix_effect = components['prefix']['effect'] if components['prefix'] else None
        
        result = {
            'executed': False,
            'output': None,
            'error': None,
            'memory_changed': False,
            'mode': self.mode.value
        }
        
        try:
            # Route to appropriate handler
            if dhatu_tag == 'print':
                result = self._execute_print(arguments, prefix_effect, suffix_mode)
            elif dhatu_tag == 'memory' or dhatu_tag == 'become':
                result = self._execute_assignment(arguments, prefix_effect, suffix_mode)
            elif dhatu_tag == 'create':
                result = self._execute_create(arguments, prefix_effect, suffix_mode)
            elif dhatu_tag == 'read':
                result = self._execute_read(arguments, prefix_effect, suffix_mode)
            elif dhatu_tag == 'calculate':
                result = self._execute_calculate(arguments, prefix_effect, suffix_mode)
            elif dhatu_tag == 'check':
                result = self._execute_check(arguments, prefix_effect, suffix_mode)
            elif dhatu_tag == 'condition':
                result = self._execute_conditional(arguments, prefix_effect, suffix_mode)
            else:
                result['output'] = f"[SIMULATION] Executed {dhatu_tag} with args: {arguments}"
                result['executed'] = True
            
            result['mode'] = self.mode.value
            
        except Exception as e:
            result['error'] = str(e)
            result['executed'] = False
            result['mode'] = self.mode.value
        
        return result
    
    def _execute_print(self, arguments: List[str], prefix: Optional[str], mode: str) -> Dict:
        """Execute print command with robust variable handling."""
        result = {'executed': True, 'memory_changed': False}
        
        if not arguments:
            result['output'] = ""
            return result
        
        # Resolve variables
        output_parts = []
        for arg in arguments:
            if self.memory.exists(arg):
                value = self.memory.get(arg)
                output_parts.append(str(value))
            else:
                output_parts.append(arg)
        
        output = ' '.join(output_parts)
        
        # Apply prefix effects with robust type handling
        if prefix == 'intensity':
            # Convert to uppercase (works for strings and string representations)
            output = str(output).upper()
        elif prefix == 'reduction':
            # Convert to lowercase
            output = str(output).lower()
        elif prefix == 'repetition':
            # Repeat based on mode
            if mode == 'imperative':
                output = '\n'.join([str(output)] * 3)  # Repeat 3 times
            else:
                output = '\n'.join([str(output)] * 2)  # Repeat 2 times for other modes
        
        result['output'] = output
        self.output_buffer.append(output)
        
        return result
    
    def _execute_assignment(self, arguments: List[str], prefix: Optional[str], mode: str) -> Dict:
        """Execute assignment/memory storage with mode-aware operators."""
        result = {'executed': True, 'memory_changed': True}
        
        try:
            var_name, value = ArgumentParser.parse_assignment(arguments, self.mode)
            
            # Apply prefix effects
            if prefix == 'intensity':
                var_name = var_name.upper()
                # Also uppercase string values
                if isinstance(value, str):
                    value = value.upper()
            elif prefix == 'reduction':
                var_name = var_name.lower()
                # Also lowercase string values
                if isinstance(value, str):
                    value = value.lower()
            elif prefix == 'collection':
                # Store as list
                if self.memory.exists(var_name):
                    existing = self.memory.get(var_name)
                    if isinstance(existing, list):
                        existing.append(value)
                        value = existing
                    else:
                        value = [existing, value]
                else:
                    value = [value]
            
            self.memory.set(var_name, value)
            result['output'] = f"Variable '{var_name}' set to {value}"
            
        except Exception as e:
            result['error'] = str(e)
            result['executed'] = False
            result['memory_changed'] = False
        
        return result
    
    def _execute_create(self, arguments: List[str], prefix: Optional[str], mode: str) -> Dict:
        """Execute create/instantiation command."""
        return self._execute_assignment(arguments, prefix, mode)
    
    def _execute_read(self, arguments: List[str], prefix: Optional[str], mode: str) -> Dict:
        """Execute read/input command."""
        result = {'executed': True, 'memory_changed': False}
        
        if mode == 'future' or mode == 'asynchronous':
            result['output'] = "[ASYNC] Waiting for input..."
        else:
            # Simulate reading input
            if arguments:
                var_name = arguments[0]
                # In real implementation, this would be input()
                result['output'] = f"[INPUT] Ready to read into '{var_name}'"
            else:
                result['output'] = "[INPUT] Ready to read"
        
        return result
    
    def _execute_calculate(self, arguments: List[str], prefix: Optional[str], mode: str) -> Dict:
        """Execute calculation command with mode-aware operators."""
        result = {'executed': True, 'memory_changed': False}
        
        if len(arguments) < 3:
            result['error'] = "Calculation requires at least 3 arguments: operand1 operator operand2"
            result['executed'] = False
            return result
        
        try:
            # Parse: var = operand1 op operand2
            if '=' in arguments or 'sama' in arguments:
                # Find assignment operator
                eq_idx = None
                for i, arg in enumerate(arguments):
                    normalized = OperatorNormalizer.normalize_operator(arg, self.mode)
                    if normalized == '=':
                        eq_idx = i
                        break
                
                if eq_idx is not None:
                    var_name = arguments[0]
                    expr_parts = arguments[eq_idx+1:]
                else:
                    var_name = None
                    expr_parts = arguments
            else:
                var_name = None
                expr_parts = arguments
            
            # Get operands (handle variables)
            operand1_str = expr_parts[0]
            if self.memory.exists(operand1_str):
                operand1 = self.memory.get(operand1_str)
            else:
                operand1 = float(operand1_str)
            
            operator = expr_parts[1]
            # Normalize operator based on mode
            normalized_op = OperatorNormalizer.normalize_operator(operator, self.mode)
            
            operand2_str = expr_parts[2]
            if self.memory.exists(operand2_str):
                operand2 = self.memory.get(operand2_str)
            else:
                operand2 = float(operand2_str)
            
            # Calculate
            if normalized_op == '+':
                calc_result = operand1 + operand2
            elif normalized_op == '-':
                calc_result = operand1 - operand2
            elif normalized_op == '*':
                calc_result = operand1 * operand2
            elif normalized_op == '/':
                calc_result = operand1 / operand2
            elif normalized_op == '%':
                calc_result = operand1 % operand2
            elif normalized_op == '**':
                calc_result = operand1 ** operand2
            else:
                raise ValueError(f"Unknown operator: {operator} (normalized: {normalized_op})")
            
            if var_name:
                self.memory.set(var_name, calc_result)
                result['memory_changed'] = True
                result['output'] = f"Calculated: {var_name} = {calc_result}"
            else:
                result['output'] = f"Result: {calc_result}"
            
        except Exception as e:
            result['error'] = str(e)
            result['executed'] = False
        
        return result
    
    def _execute_check(self, arguments: List[str], prefix: Optional[str], mode: str) -> Dict:
        """Execute check/validation command."""
        result = {'executed': True, 'memory_changed': False}
        
        if not arguments:
            result['output'] = "[CHECK] No argument to check"
            return result
        
        var_name = arguments[0]
        if self.memory.exists(var_name):
            value = self.memory.get(var_name)
            result['output'] = f"✓ Variable '{var_name}' exists with value: {value}"
        else:
            result['output'] = f"✗ Variable '{var_name}' does not exist"
        
        return result
    
    def _execute_conditional(self, arguments: List[str], prefix: Optional[str], mode: str) -> Dict:
        """Execute conditional command with mode-aware operators."""
        result = {'executed': True, 'memory_changed': False}
        
        try:
            # Parse condition and action
            condition_tokens, action_tokens = ConditionalParser.parse_conditional(arguments)
            
            # Evaluate condition with mode-aware operators
            condition_result = ConditionalParser.evaluate_condition(condition_tokens, self.memory, self.mode)
            
            if condition_result:
                # Condition is true - execute action
                # The action is another full command, so we need to recursively process it
                action_command = ' '.join(action_tokens)
                result['output'] = f"[CONDITIONAL] Condition TRUE: {' '.join(condition_tokens)}\n"
                result['output'] += f"[CONDITIONAL] Executing: {action_command}"
                result['conditional_result'] = True
                result['action_command'] = action_command
            else:
                # Condition is false - skip action
                result['output'] = f"[CONDITIONAL] Condition FALSE: {' '.join(condition_tokens)}\n"
                result['output'] += f"[CONDITIONAL] Skipping action"
                result['conditional_result'] = False
            
        except Exception as e:
            result['error'] = str(e)
            result['executed'] = False
        
        return result
    
    def get_output_buffer(self) -> List[str]:
        """Get all output."""
        return self.output_buffer.copy()
    
    def clear_output(self):
        """Clear output buffer."""
        self.output_buffer.clear()


class EasyPeasyHindiCombinator:
    """Enhanced logic resolver with dual-mode execution capabilities."""
    
    def __init__(self, json_data: Dict, mode: ExecutionMode = ExecutionMode.HYBRID):
        """Initialize the combinator with specified mode."""
        self.dhatus = {item['transliteration']: item for item in json_data['dhatus']}
        self.upasargas = {item['transliteration']: item for item in json_data['upasargas']}
        self.pratyayas = {item['transliteration']: item for item in json_data['pratyayas']}
        
        # Mode management
        self.mode = mode
        
        # Components
        self.memory = GlobalMemory()
        self.execution_engine = ExecutionEngine(self.memory, self.mode)
        self.arg_parser = ArgumentParser()
        self.normalizer = ASCIINormalizer()
        self.operator_normalizer = OperatorNormalizer()
        self.statement_validator = StatementValidator()
    
    def set_mode(self, mode: ExecutionMode):
        """Switch execution mode."""
        self.mode = mode
        self.execution_engine.set_mode(mode)
    
    def get_mode(self) -> ExecutionMode:
        """Get current execution mode."""
        return self.mode
    
    def parse_command(self, command: str) -> Tuple[Optional[str], str, str]:
        """Parse command into components with ASCII normalization."""
        # Step 1: Normalize ASCII input to Sanskrit transliteration
        normalized_command = self.normalizer.normalize_command(command)
        
        # Step 2: Parse normalized command
        parts = normalized_command.strip().split('-')
        
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            return None, parts[0], parts[1]
        else:
            raise ValueError(f"Invalid command format: '{command}'. Expected 'prefix-dhatu-suffix' or 'dhatu-suffix'")
    
    def resolve_components(self, prefix: Optional[str], dhatu: str, suffix: str) -> Dict:
        """Resolve components from database."""
        result = {}
        
        if prefix:
            if prefix not in self.upasargas:
                raise KeyError(f"Prefix '{prefix}' not found in upasargas database")
            result['prefix'] = self.upasargas[prefix]
        else:
            result['prefix'] = None
        
        if dhatu not in self.dhatus:
            raise KeyError(f"Dhatu '{dhatu}' not found in dhatus database")
        result['dhatu'] = self.dhatus[dhatu]
        
        if suffix not in self.pratyayas:
            raise KeyError(f"Suffix '{suffix}' not found in pratyayas database")
        result['suffix'] = self.pratyayas[suffix]
        
        return result
    
    def generate_logic(self, components: Dict) -> Dict:
        """Generate programming logic description."""
        prefix_data = components['prefix']
        dhatu_data = components['dhatu']
        suffix_data = components['suffix']
        
        execution_mode = suffix_data['mode'].upper()
        execution_type = suffix_data['execution']
        dhatu_action = dhatu_data['tag'].upper()
        prefix_effect = prefix_data['effect'].upper() if prefix_data else "DIRECT"
        
        if prefix_data:
            logic_description = f"{execution_mode} {prefix_effect} {dhatu_action}"
            detailed_description = (
                f"{suffix_data['description']} with {prefix_data['description']} "
                f"applied to {dhatu_data['description']}"
            )
        else:
            logic_description = f"{execution_mode} {dhatu_action}"
            detailed_description = (
                f"{suffix_data['description']} for {dhatu_data['description']}"
            )
        
        return {
            'logic': logic_description,
            'detailed': detailed_description,
            'execution_type': execution_type,
            'components': {
                'prefix': prefix_data['prefix'] if prefix_data else None,
                'dhatu': dhatu_data['dhatu'],
                'suffix': suffix_data['suffix']
            },
            'devanagari_form': self._build_devanagari(components)
        }
    
    def _build_devanagari(self, components: Dict) -> str:
        """Build Devanagari representation."""
        prefix_dev = components['prefix']['prefix'] if components['prefix'] else ''
        dhatu_dev = components['dhatu']['dhatu']
        suffix_dev = components['suffix']['suffix']
        return f"{prefix_dev}{dhatu_dev}{suffix_dev}"
    
    def process_command(self, input_string: str, execute: bool = True) -> Dict:
        """
        Main processing pipeline with mode-aware execution.
        
        Args:
            input_string: Full input string with command and arguments
            execute: Whether to actually execute the command
        """
        try:

            cleaned_input, is_script_end = self.statement_validator.validate_statement(
                input_string, self.mode
            )
            
            command, arguments = self.arg_parser.extract_arguments(cleaned_input)
            
            prefix, dhatu, suffix = self.parse_command(command)
            
            components = self.resolve_components(prefix, dhatu, suffix)
            
            logic = self.generate_logic(components)
            
            exec_result = None
            if execute:
                if suffix == 'et':
                    exec_result = self._handle_conditional_execution(components, arguments)
                else:
                    exec_result = self.execution_engine.execute(components, arguments)
            
            result = {
                'status': 'success',
                'input': input_string,
                'command': command,
                'normalized_command': f"{prefix + '-' if prefix else ''}{dhatu}-{suffix}",
                'arguments': arguments,
                'mode': self.mode.value,
                'is_script_end': is_script_end,
                **logic
            }
            
            if exec_result:
                result['execution'] = exec_result
            
            return result
            
        except (ValueError, KeyError) as e:
            return {
                'status': 'error',
                'input': input_string,
                'error': str(e),
                'message': 'Failed to process command',
                'mode': self.mode.value
            }
    
    def _handle_conditional_execution(self, components: Dict, arguments: List[str]) -> Dict:
        """Handle conditional execution for -et suffix."""
        result = self.execution_engine.execute(components, arguments)
        
        if result.get('executed') and result.get('conditional_result') and 'action_command' in result:
            action_command = result['action_command']
            
            if self.mode == ExecutionMode.SHUDDH:
                action_command = StatementValidator.add_terminator(action_command, self.mode)
            
            action_result = self.process_command(action_command, execute=True)
            
            if action_result['status'] == 'success' and 'execution' in action_result:
                action_exec = action_result['execution']
                result['action_execution'] = action_exec
                
#outputs merged
                if action_exec.get('output'):
                    result['output'] += f"\n{action_exec['output']}"
                
                if action_exec.get('memory_changed'):
                    result['memory_changed'] = True
        
        return result
    
    def print_result(self, result: Dict):
        """Pretty print the result."""
        print("\n" + "="*70)
        
        if result['status'] == 'success':
            print(f"✓ INPUT: {result['input']}")
            print(f"✓ MODE: {result['mode'].upper()}")
            print(f"✓ COMMAND: {result['command']}")
            if result['command'] != result.get('normalized_command', ''):
                print(f"✓ NORMALIZED: {result['normalized_command']}")
            if result['arguments']:
                print(f"✓ ARGUMENTS: {result['arguments']}")
            print(f"✓ DEVANAGARI: {result['devanagari_form']}")
            if result.get('is_script_end'):
                print(f"✓ SCRIPT END: || (Deergha Virama detected)")
            print("-"*70)
            print(f"LOGIC: {result['logic']}")
            print(f"EXECUTION TYPE: {result['execution_type']}")
            
            # Show execution results
            if 'execution' in result:
                exec_res = result['execution']
                print("-"*70)
                print("EXECUTION RESULT:")
                if exec_res['executed']:
                    print(f"  ✓ Status: SUCCESS")
                    if exec_res['output']:
                        print(f"  ✓ Output:\n{exec_res['output']}")
                    if exec_res['memory_changed']:
                        print(f"  ✓ Memory: MODIFIED")
                    
                    # Show action execution for conditionals 
                    if 'action_execution' in exec_res:
                        print(f"\n  Action Execution:")
                        action_exec = exec_res['action_execution']
                        if action_exec.get('output'):
                            print(f"    Output: {action_exec['output']}")
                else:
                    print(f"  ✗ Status: FAILED")
                    if exec_res['error']:
                        print(f"  ✗ Error: {exec_res['error']}")
        else:
            print(f"✗ INPUT: {result['input']}")
            print(f"✗ MODE: {result['mode'].upper()}")
            print(f"✗ ERROR: {result['error']}")
        
        print("="*70)
    
    def show_memory(self):
        """Display current mem. state"""
        print("\n" + "="*70)
        print("  GLOBAL MEMORY STATE")
        print("="*70)
        
        variables = self.memory.list_all()
        
        if not variables:
            print("  (empty)")
        else:
            for name, value in variables.items():
                print(f"  {name} = {value} ({type(value).__name__})")
        
        print("="*70)
    
    def clear_memory(self):
        """Clear all memory."""
        self.memory.clear()
        self.execution_engine.clear_output()
        print("\n✓ Memory cleared")
    
    def show_mappings(self):
        """Show ASCII to Sanskrit mappings."""
        print("\n" + "="*70)
        print(self.normalizer.get_mapping_info())
        print("="*70)
    
    def show_operators(self):
        """Show operator mappings."""
        print("\n" + "="*70)
        print(self.operator_normalizer.get_operator_info())
        print("="*70)
    
    def show_mode_info(self):
        """Show current mode and its requirements."""
        print("\n" + "="*70)
        print(f"  CURRENT MODE: {self.mode.value.upper()}")
        print("="*70)
        
        if self.mode == ExecutionMode.HYBRID:
            print("\nHYBRID Mode:")
            print("  • Standard symbols allowed: =, +, -, *, /, <, >, ==, etc.")
            print("  • Sanskrit operators also work: sama, yuj, viyuj, etc.")
            print("  • Statement terminator (|) optional")
            print("\nExample:")
            print("  bhu-ati x = 10")
            print("  gan-atu sum = x + 5")
        else:
            print("\nSHUDDH Mode (Pure Sanskrit):")
            print("  • MUST use Sanskrit operators: sama, yuj, viyuj, adhika, etc.")
            print("  • MUST end statements with | (Purna Virama)")
            print("  • || (Deergha Virama) signals script end")
            print("\nExample:")
            print("  bhu-ati x sama 10 |")
            print("  gan-atu sum sama x yuj 5 |")
            print("  man-et x adhika 5 vad-atu Big ||")
        
        print("="*70)


def interactive_mode(combinator: EasyPeasyHindiCombinator):
    """Run interactive testing mode with dual-mode support."""
    print("\n" + "="*70)
    print("  EasyPeasyHindi Combinator v2.6 - Dual-Mode Execution")
    print("  HYBRID & SHUDDH Modes with Sanskrit Operators!")
    print("="*70)
    print(f"\nCurrent Mode: {combinator.get_mode().value.upper()}")
    print("\nFormat: command arguments [|]")
    print("\nHYBRID Mode Examples:")
    print("  bhu-ati x = 10")
    print("  gan-atu sum = x + 5")
    print("  man-et x > 5 vad-atu Big")
    print("\nSHUDDH Mode Examples (requires | terminator):")
    print("  bhu-ati x sama 10 |")
    print("  gan-atu sum sama x yuj 5 |")
    print("  man-et x adhika 5 vad-atu Big |")
    print("\nSpecial commands:")
    print("  :mode hybrid   - Switch to HYBRID mode")
    print("  :mode shuddh   - Switch to SHUDDH mode (pure Sanskrit)")
    print("  :mode          - Show current mode info")
    print("  :operators     - Show Sanskrit operator mappings")
    print("  :memory        - Show current memory state")
    print("  :clear         - Clear all memory")
    print("  :mappings      - Show ASCII→Sanskrit mappings")
    print("  :quit          - Exit program")
    print()
    
    while True:
        try:
            mode_indicator = "SHUDDH" if combinator.get_mode() == ExecutionMode.SHUDDH else "HYBRID"
            user_input = input(f"EasyPeasy[{mode_indicator}]> ").strip()
            
            if not user_input:
                continue
            
            # Special commands
            if user_input.startswith(':'):
                if user_input in [':quit', ':exit', ':q']:
                    print("\n👋 Namaste! Exiting...")
                    break
                elif user_input in [':memory', ':mem']:
                    combinator.show_memory()
                    continue
                elif user_input == ':clear':
                    combinator.clear_memory()
                    continue
                elif user_input in [':mappings', ':map']:
                    combinator.show_mappings()
                    continue
                elif user_input in [':operators', ':ops']:
                    combinator.show_operators()
                    continue
                elif user_input == ':mode':
                    combinator.show_mode_info()
                    continue
                elif user_input == ':mode hybrid':
                    combinator.set_mode(ExecutionMode.HYBRID)
                    print(f"\n✓ Mode switched to: HYBRID")
                    print("  Standard symbols (=, +, -) now allowed")
                    print("  Statement terminator (|) optional")
                    continue
                elif user_input == ':mode shuddh':
                    combinator.set_mode(ExecutionMode.SHUDDH)
                    print(f"\n✓ Mode switched to: SHUDDH")
                    print("  Pure Sanskrit operators required (sama, yuj, viyuj)")
                    print("  Statements MUST end with | (Purna Virama)")
                    continue
                elif user_input == ':help':
                    print("\nAvailable commands:")
                    print("  :mode [hybrid|shuddh], :operators, :memory, :clear,")
                    print("  :mappings, :quit, :help")
                    continue
                else:
                    print(f"Unknown command: {user_input}")
                    continue
            
            # Process and execute
            result = combinator.process_command(user_input, execute=True)
            combinator.print_result(result)
            
            # Check for script end signal
            if result.get('is_script_end'):
                print("\n⚡ Deergha Virama (||) detected - Script execution complete")
                print("   (Continuing in interactive mode...)")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Namaste!")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")


def main():
    """Main entry point."""
    json_database = {
        "dhatus": [
            {"dhatu": "वद्", "transliteration": "vad", "tag": "print", "description": "Output/Display data"},
            {"dhatu": "कृ", "transliteration": "kṛ", "tag": "create", "description": "Create variables or objects"},
            {"dhatu": "स्था", "transliteration": "sthā", "tag": "memory", "description": "Store data in memory"},
            {"dhatu": "भू", "transliteration": "bhū", "tag": "become", "description": "Transform or assign"},
            {"dhatu": "पठ्", "transliteration": "paṭh", "tag": "read", "description": "Read input"},
            {"dhatu": "गण्", "transliteration": "gaṇ", "tag": "calculate", "description": "Perform calculations"},
            {"dhatu": "दृश्", "transliteration": "dṛś", "tag": "check", "description": "Verify or inspect"},
            {"dhatu": "चल्", "transliteration": "cal", "tag": "run", "description": "Execute or run"},
            {"dhatu": "मन्", "transliteration": "man", "tag": "condition", "description": "Conditional check"}
        ],
        "upasargas": [
            {"prefix": "प्र", "transliteration": "pra", "effect": "intensity", "description": "High priority or uppercase"},
            {"prefix": "अनु", "transliteration": "anu", "effect": "repetition", "description": "Loop or iterate"},
            {"prefix": "प्रति", "transliteration": "prati", "effect": "reaction", "description": "Callback or event"},
            {"prefix": "वि", "transliteration": "vi", "effect": "distinction", "description": "Special case"},
            {"prefix": "सम्", "transliteration": "sam", "effect": "collection", "description": "Aggregate or combine"},
            {"prefix": "अव", "transliteration": "ava", "effect": "reduction", "description": "Minimize or lowercase"}
        ],
        "pratyayas": [
            {"suffix": "अतु", "transliteration": "atu", "mode": "imperative", "execution": "synchronous", "description": "Execute immediately"},
            {"suffix": "अति", "transliteration": "ati", "mode": "indicative", "execution": "declarative", "description": "State declaration"},
            {"suffix": "अणे", "transliteration": "aṇe", "mode": "future", "execution": "asynchronous", "description": "Async operation"},
            {"suffix": "एत्", "transliteration": "et", "mode": "conditional", "execution": "conditional", "description": "Conditional execution"}
        ]
    }
    
    # Start in HYBRID mode by default
    combinator = EasyPeasyHindiCombinator(json_database, mode=ExecutionMode.HYBRID)
    
    if len(sys.argv) > 1:
        # Command line execution
        full_command = ' '.join(sys.argv[1:])
        result = combinator.process_command(full_command, execute=True)
        combinator.print_result(result)
    else:
        interactive_mode(combinator)


if _name__ == "__main__":
    main()
