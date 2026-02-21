#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EasyPeasyHindi v2.6
A small, playful command engine inspired by Sanskrit grammar.
"""

import sys
import re
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple


# -------------------------------------------------------------------
# Execution Mode
# -------------------------------------------------------------------

class ExecutionMode(Enum):
    HYBRID = "hybrid"   # symbols + Sanskrit
    SHUDDH = "shuddh"   # Sanskrit only


# -------------------------------------------------------------------
# Operator Handling
# -------------------------------------------------------------------

class OperatorNormalizer:
    """Handles Sanskrit ↔ symbol operator translation."""

    ASSIGN = {"sama": "="}
    COMPARE = {
        "tulya": "==",
        "adhika": ">",
        "nyuna": "<",
        "adhika-tulya": ">=",
        "nyuna-tulya": "<=",
        "na-tulya": "!=",
    }
    MATH = {
        "yuj": "+",
        "viyuj": "-",
        "gun": "*",
        "bhaj": "/",
        "shesh": "%",
        "ghat": "**",
    }

    ALL = {**ASSIGN, **COMPARE, **MATH}
    REVERSE = {v: k for k, v in ALL.items()}

    @staticmethod
    def normalize(op: str, mode: ExecutionMode) -> str:
        return OperatorNormalizer.ALL.get(op, op) if mode == ExecutionMode.SHUDDH else op

    @staticmethod
    def is_valid(op: str, mode: ExecutionMode) -> bool:
        if mode == ExecutionMode.SHUDDH:
            return op in OperatorNormalizer.ALL
        return op in OperatorNormalizer.ALL or op in OperatorNormalizer.REVERSE

    @staticmethod
    def info() -> str:
        lines = ["Sanskrit Operator Mappings:"]
        for label, group in [
            ("Assignment", OperatorNormalizer.ASSIGN),
            ("Comparison", OperatorNormalizer.COMPARE),
            ("Arithmetic", OperatorNormalizer.MATH),
        ]:
            lines.append(f"\n{label}:")
            for k, v in group.items():
                lines.append(f"  {k:15} → {v}")
        return "\n".join(lines)


# -------------------------------------------------------------------
# Statement Handling
# -------------------------------------------------------------------

class StatementValidator:
    """Handles | and || terminators."""

    @staticmethod
    def validate(line: str, mode: ExecutionMode) -> Tuple[str, bool]:
        text = line.strip()

        if text.endswith("||"):
            return text[:-2].strip(), True

        if text.endswith("|"):
            return text[:-1].strip(), False

        if mode == ExecutionMode.SHUDDH:
            raise ValueError("SHUDDH mode requires '|' terminator")

        return text, False

    @staticmethod
    def add(statement: str, mode: ExecutionMode, end=False) -> str:
        if mode != ExecutionMode.SHUDDH:
            return statement
        return f"{statement} ||" if end else f"{statement} |"


# -------------------------------------------------------------------
# ASCII → Sanskrit Normalization
# -------------------------------------------------------------------

class ASCIINormalizer:
    """Normalizes easy typing into proper transliteration."""

    DHATU = {
        "kr": "kṛ",
        "stha": "sthā",
        "bhu": "bhū",
        "path": "paṭh",
        "gan": "gaṇ",
        "drs": "dṛś",
        "drsh": "dṛś",
    }

    UPASARGA = {"pra", "anu", "prati", "vi", "sam", "ava"}
    PRATYAYA = {"ane": "aṇe"}

    @staticmethod
    def normalize(cmd: str) -> str:
        parts = cmd.split("-")

        def map_part(p, table):
            return table.get(p, p)

        if len(parts) == 3:
            return "-".join([
                map_part(parts[0], ASCIINormalizer.UPASARGA),
                map_part(parts[1], ASCIINormalizer.DHATU),
                map_part(parts[2], ASCIINormalizer.PRATYAYA),
            ])

        if len(parts) == 2:
            return "-".join([
                map_part(parts[0], ASCIINormalizer.DHATU),
                map_part(parts[1], ASCIINormalizer.PRATYAYA),
            ])

        return cmd


# -------------------------------------------------------------------
# Memory Store
# -------------------------------------------------------------------

class GlobalMemory:
    """history ke saath chhoti key-value memory"""

    def __init__(self):
        self.data = {}
        self.history = []

    def set(self, key, value):
        self.data[key] = value
        self.history.append(("SET", key, value))

    def get(self, key):
        if key not in self.data:
            raise KeyError(f"Variable '{key}' not found")
        return self.data[key]

    def exists(self, key):
        return key in self.data

    def clear(self):
        self.data.clear()
        self.history.clear()

    def dump(self):
        return self.data.copy()


# -------------------------------------------------------------------
# Argument Parsing
# -------------------------------------------------------------------

class ArgumentParser:
    """Splits command + arguments."""

    @staticmethod
    def extract(line: str) -> Tuple[str, List[str]]:
        tokens = re.findall(r'"[^"]*"|\S+', line)
        if not tokens:
            raise ValueError("Empty input")

        cmd = tokens[0]
        args = [t[1:-1] if t.startswith('"') else t for t in tokens[1:]]
        return cmd, args

    @staticmethod
    def parse_assignment(args: List[str], mode: ExecutionMode):
        if len(args) < 3:
            raise ValueError("Invalid assignment")

        var, op = args[0], args[1]
        op = OperatorNormalizer.normalize(op, mode)

        if op != "=":
            raise ValueError("Invalid assignment operator")

        raw = " ".join(args[2:])
        try:
            return var, int(raw) if raw.isdigit() else float(raw)
        except ValueError:
            return var, raw.strip('"')


# -------------------------------------------------------------------
# Conditionals
# -------------------------------------------------------------------

class ConditionalParser:

    @staticmethod
    def split(args: List[str]):
        for i, a in enumerate(args):
            if "-" in a:
                return args[:i], args[i:]
        raise ValueError("Invalid conditional")

    @staticmethod
    def parse_value(v):
        try:
            return int(v) if v.isdigit() else float(v)
        except ValueError:
            return v.strip('"')

    @staticmethod
    def evaluate(tokens, memory: GlobalMemory, mode: ExecutionMode):
        left, op, right = tokens[:3]

        left = memory.get(left) if memory.exists(left) else ConditionalParser.parse_value(left)
        right = memory.get(right) if memory.exists(right) else ConditionalParser.parse_value(right)
        op = OperatorNormalizer.normalize(op, mode)

        return {
            "==": left == right,
            "!=": left != right,
            "<": left < right,
            ">": left > right,
            "<=": left <= right,
            ">=": left >= right,
        }.get(op, False)


# -------------------------------------------------------------------
# Execution Engine
# -------------------------------------------------------------------

class ExecutionEngine:
    """Executes resolved commands."""

    def __init__(self, memory: GlobalMemory, mode=ExecutionMode.HYBRID):
        self.memory = memory
        self.mode = mode
        self.output = []

    def set_mode(self, mode):
        self.mode = mode

    def execute(self, components, args):
        tag = components["dhatu"]["tag"]
        handler = getattr(self, f"_do_{tag}", None)

        if not handler:
            return {"executed": True, "output": f"[SIMULATION] {tag} {args}"}

        return handler(args, components)

    def _do_print(self, args, comp):
        out = []
        for a in args:
            out.append(str(self.memory.get(a)) if self.memory.exists(a) else a)

        text = " ".join(out)
        self.output.append(text)
        return {"executed": True, "output": text}

    def _do_become(self, args, comp):
        var, val = ArgumentParser.parse_assignment(args, self.mode)
        self.memory.set(var, val)
        return {"executed": True, "memory_changed": True, "output": f"{var} = {val}"}


# -------------------------------------------------------------------
# Core Combinator
# -------------------------------------------------------------------

class EasyPeasyHindiCombinator:
    """Main orchestration class."""

    def __init__(self, db: Dict, mode=ExecutionMode.HYBRID):
        self.mode = mode
        self.memory = GlobalMemory()
        self.engine = ExecutionEngine(self.memory, mode)

        self.dhatus = {d["transliteration"]: d for d in db["dhatus"]}
        self.upasargas = {u["transliteration"]: u for u in db["upasargas"]}
        self.pratyayas = {p["transliteration"]: p for p in db["pratyayas"]}

    def set_mode(self, mode):
        self.mode = mode
        self.engine.set_mode(mode)

    def process(self, line: str):
        clean, end = StatementValidator.validate(line, self.mode)
        cmd, args = ArgumentParser.extract(clean)

        cmd = ASCIINormalizer.normalize(cmd)
        parts = cmd.split("-")

        prefix = self.upasargas.get(parts[0]) if len(parts) == 3 else None
        dhatu = self.dhatus[parts[-2]]
        suffix = self.pratyayas[parts[-1]]

        components = {"prefix": prefix, "dhatu": dhatu, "suffix": suffix}
        result = self.engine.execute(components, args)
        result["is_script_end"] = end
        return result


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

def main():
    # Same JSON DB as original (unchanged)
    from json import loads
    db = loads(open(__file__).read().split("json_database =")[1].split("if __name__")[0])

    engine = EasyPeasyHindiCombinator(db)

    if len(sys.argv) > 1:
        print(engine.process(" ".join(sys.argv[1:])))
    else:
        while True:
            try:
                line = input("EasyPeasy> ")
                if line in (":q", ":quit"):
                    break
                print(engine.process(line))
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
