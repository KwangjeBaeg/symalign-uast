"""
Obfuscation Module for Experiment 02.

Five conditions operating on code-like text (library import lines + comments):
  C0: Clean (0% obfuscation)
  C1: Import masking only — library/package names replaced
  C2: Identifier stripping only — variable/function aliases replaced
  C3: Partial obfuscation (50% of unique tokens replaced)
  C4: Full obfuscation (100% of unique tokens replaced)
"""

import json
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple


def _random_token(rng: random.Random, prefix: str = "TOK", length: int = 8) -> str:
    chars = string.ascii_lowercase + string.digits
    return f"{prefix}_" + "".join(rng.choices(chars, k=length))


# ── Text reconstruction from UAST ──

def reconstruct_code_text(file_id: str, uast_dir: str, libraries: List[str],
                          comments: List[str]) -> str:
    """
    Build a code-like text from UAST data for a single file.

    Strategy:
      1. Try to load the UAST JSON and extract import lines + comments.
      2. Fall back to synthetic import lines from the library list.

    The result is a multi-line string that pretrained code models can consume.
    """
    lines: List[str] = []

    # Try UAST JSON for richer text
    uast_path = Path(uast_dir) / f"UAST_{file_id}.json"
    if uast_path.exists():
        try:
            with open(uast_path) as f:
                uast = json.load(f)
            nodes = uast.get("nodes", {})
            for nid in sorted(nodes.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                node = nodes[nid]
                ntype = node.get("node_type", "")
                snippet = node.get("code_snippet", "").strip()
                if not snippet or snippet == "root":
                    continue
                if "comment" in ntype:
                    # Strip "uast_comment" prefix
                    text = snippet.replace("uast_comment", "").strip()
                    if text:
                        lines.append(f"# {text}")
                elif "package" in ntype or "import" in ntype:
                    lib = snippet.replace("uast_package", "").replace("uast_import", "").strip()
                    if lib:
                        lines.append(f"import {lib}")
                elif "function" in ntype or "class" in ntype:
                    name = snippet.replace(f"uast_{ntype.split('_')[-1]}", "").strip()
                    if name:
                        lines.append(f"def {name}(): pass")
        except Exception:
            pass

    # Fall back / augment with library list
    if not lines:
        for lib in libraries:
            lines.append(f"import {lib}")
        for c in (comments or []):
            lines.append(f"# {c}")

    return "\n".join(lines) if lines else " ".join(libraries)


def build_code_texts(dataset: List[Dict], uast_dir: str) -> List[str]:
    """Build code-like text for every file in the dataset."""
    texts = []
    for item in dataset:
        text = reconstruct_code_text(
            file_id=item["file_id"],
            uast_dir=uast_dir,
            libraries=item.get("all_libraries", item["libraries"]),
            comments=item.get("comments", []),
        )
        texts.append(text)
    return texts


# ── Obfuscation transforms ──

def _collect_library_tokens(texts: List[str], libraries_flat: List[str]) -> List[str]:
    """Unique library/package names across all texts."""
    lib_set = set(libraries_flat)
    return sorted(lib_set)


def _collect_identifier_tokens(texts: List[str], library_tokens: set) -> List[str]:
    """
    Collect non-library code identifiers (function/variable names only).

    Only extracts tokens from code lines (import/def/class), NOT from comments.
    This keeps the token set manageable (<1000 vs 100K+ with comments).
    """
    idents = set()
    keywords = {"import", "from", "def", "class", "pass", "return", "if",
                "else", "for", "while", "try", "except", "with", "as", "in",
                "not", "and", "or", "is", "None", "True", "False", "#"}
    for text in texts:
        for line in text.split("\n"):
            # Skip comment lines — they contain too many English words
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            tokens = stripped.split()
            for tok in tokens:
                tok_clean = tok.strip("():,")
                if (tok_clean and tok_clean not in library_tokens
                        and tok_clean not in keywords
                        and len(tok_clean) >= 2):
                    idents.add(tok_clean)
    return sorted(idents)


def _apply_mapping_fast(texts: List[str], mapping: Dict[str, str]) -> List[str]:
    """
    Efficiently apply token mapping via line-by-line word splitting.

    Instead of O(tokens × texts × text_length) string.replace() calls,
    this splits each line into words and replaces in O(words) per line.
    """
    if not mapping:
        return list(texts)
    import re
    # Pattern: split into (non-word prefix)(word core)(non-word suffix)
    _tok_re = re.compile(r'^([^a-zA-Z0-9_]*)(.*?)([^a-zA-Z0-9_]*)$')
    out = []
    for text in texts:
        new_lines = []
        for line in text.split("\n"):
            tokens = line.split()
            new_tokens = []
            for tok in tokens:
                m = _tok_re.match(tok)
                prefix, core, suffix = m.group(1), m.group(2), m.group(3)
                replaced = mapping.get(core, core)
                new_tokens.append(prefix + replaced + suffix)
            new_lines.append(" ".join(new_tokens))
        out.append("\n".join(new_lines))
    return out


def obfuscate_c1(texts: List[str], library_tokens: List[str],
                 seed: int = 42) -> Tuple[List[str], Dict[str, str]]:
    """C1: Import masking only — replace library names."""
    rng = random.Random(seed)
    mapping = {lib: _random_token(rng, "LIB") for lib in library_tokens}
    return _apply_mapping_fast(texts, mapping), mapping


def obfuscate_c2(texts: List[str], library_tokens: List[str],
                 seed: int = 42) -> Tuple[List[str], Dict[str, str]]:
    """C2: Identifier stripping only — replace non-library identifiers."""
    rng = random.Random(seed)
    lib_set = set(library_tokens)
    idents = _collect_identifier_tokens(texts, lib_set)
    mapping = {ident: _random_token(rng, "ID") for ident in idents}
    return _apply_mapping_fast(texts, mapping), mapping


def obfuscate_c3(texts: List[str], all_unique_tokens: List[str],
                 seed: int = 42) -> Tuple[List[str], Dict[str, str]]:
    """C3: 50% of unique tokens replaced."""
    rng = random.Random(seed)
    n = len(all_unique_tokens) // 2
    selected = rng.sample(all_unique_tokens, n)
    mapping = {tok: _random_token(rng, "OBF") for tok in selected}
    return _apply_mapping_fast(texts, mapping), mapping


def obfuscate_c4(texts: List[str], all_unique_tokens: List[str],
                 seed: int = 42) -> Tuple[List[str], Dict[str, str]]:
    """C4: 100% of unique tokens replaced."""
    rng = random.Random(seed)
    mapping = {tok: _random_token(rng, "OBF") for tok in all_unique_tokens}
    return _apply_mapping_fast(texts, mapping), mapping


def apply_obfuscation(condition: str, texts: List[str],
                      library_tokens: List[str],
                      all_unique_tokens: List[str],
                      seed: int = 42) -> Tuple[List[str], Dict[str, str]]:
    """
    Apply obfuscation condition to texts.

    Args:
        condition: one of "C0", "C1", "C2", "C3", "C4"
        texts: list of code-like strings
        library_tokens: unique library names
        all_unique_tokens: union of library + identifier tokens
        seed: random seed

    Returns:
        obfuscated texts, mapping dict
    """
    if condition == "C0":
        return list(texts), {}
    elif condition == "C1":
        return obfuscate_c1(texts, library_tokens, seed)
    elif condition == "C2":
        return obfuscate_c2(texts, library_tokens, seed)
    elif condition == "C3":
        return obfuscate_c3(texts, all_unique_tokens, seed)
    elif condition == "C4":
        return obfuscate_c4(texts, all_unique_tokens, seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")
