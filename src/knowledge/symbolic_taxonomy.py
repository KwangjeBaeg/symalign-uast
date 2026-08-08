"""
Symbolic Taxonomy: Abstract Behavioral Categories for UAST

Maps thousands of API calls found in UAST code_snippets into 7 abstract
behavioral categories. This is NOT identifier-level learning — it derives
logical properties (e.g., "this code performs data I/O") from code patterns,
which are then used to verify GNN predictions via symbolic logic losses.

The taxonomy is designed to be:
  - Abstract: categories represent high-level behaviors, not specific APIs
  - Exhaustive: covers the major behavioral patterns in the 14 domain classes
  - Discriminative: different domains exhibit different behavioral signatures
"""

import re
from typing import Dict, List, Set


# =============================================================================
# 7 Abstract Behavioral Categories
# =============================================================================

BEHAVIORAL_CATEGORIES = [
    "data_io",
    "math_compute",
    "network_comm",
    "visualization",
    "model_training",
    "text_processing",
    "system_ops",
]

NUM_BEHAVIORAL_CATEGORIES = len(BEHAVIORAL_CATEGORIES)
CATEGORY_TO_IDX = {cat: i for i, cat in enumerate(BEHAVIORAL_CATEGORIES)}


# =============================================================================
# Keyword Taxonomy per Category
#
# Two types of evidence:
#   1. call_keywords: patterns matched in uast_call node code_snippets
#   2. package_keywords: patterns matched in uast_package original_code
# =============================================================================

TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "data_io": {
        "call_keywords": [
            "open", "read_csv", "to_csv", "read_json", "to_json",
            "read_excel", "to_excel", "read_parquet", "to_parquet",
            "read_sql", "read_hdf", "read_pickle", "to_pickle",
            "load", "save", "dump", "loads", "dumps",
            "DataFrame", "from_dict", "from_records",
            "read_file", "write_file", "readlines", "writelines",
            "csv.reader", "csv.writer", "DictReader", "DictWriter",
            "h5py", "File",
        ],
        "package_keywords": [
            "csv", "json", "pickle", "h5py", "pandas",
            "openpyxl", "xlrd", "pyarrow", "parquet",
            "yaml", "toml", "configparser", "io",
        ],
    },
    "math_compute": {
        "call_keywords": [
            "dot", "matmul", "multiply", "divide",
            "array", "ndarray", "zeros", "ones", "linspace", "arange",
            "sin", "cos", "tan", "exp", "log", "sqrt", "abs",
            "mean", "std", "var", "sum", "prod", "cumsum",
            "linalg", "solve", "inv", "det", "eig", "svd", "norm",
            "fft", "ifft", "integrate", "diff", "gradient",
            "random", "rand", "randn", "uniform", "normal",
            "matrix", "tensor", "sparse",
        ],
        "package_keywords": [
            "numpy", "scipy", "sympy", "math", "cmath",
            "statistics", "decimal", "fractions",
            "linalg", "optimize", "interpolate",
        ],
    },
    "network_comm": {
        "call_keywords": [
            "request", "get", "post", "put", "delete", "patch",
            "http", "https", "url", "uri",
            "socket", "connect", "bind", "listen", "accept",
            "send", "recv", "sendall",
            "session", "response", "status_code",
            "route", "endpoint", "api",
            "fetch", "download", "upload",
            "websocket", "grpc",
        ],
        "package_keywords": [
            "requests", "urllib", "http", "httpx",
            "flask", "django", "fastapi", "aiohttp",
            "socket", "socketio", "websocket",
            "grpc", "tornado", "bottle", "starlette",
        ],
    },
    "visualization": {
        "call_keywords": [
            "plot", "scatter", "bar", "hist", "boxplot",
            "figure", "subplot", "subplots", "axes",
            "xlabel", "ylabel", "title", "legend",
            "savefig", "show", "imshow", "contour",
            "draw", "render", "canvas", "colorbar",
            "heatmap", "pairplot", "distplot",
            "set_xlim", "set_ylim", "annotate",
        ],
        "package_keywords": [
            "matplotlib", "plotly", "seaborn", "bokeh",
            "altair", "pygal", "graphviz", "pillow",
            "PIL", "cv2", "imageio",
        ],
    },
    "model_training": {
        "call_keywords": [
            "fit", "predict", "predict_proba", "transform",
            "fit_transform", "score", "evaluate",
            "train", "compile", "build",
            "loss", "optimizer", "backward", "step",
            "gradient", "epoch", "batch",
            "Sequential", "Dense", "Conv2d", "Linear",
            "DataLoader", "Dataset",
            "accuracy", "precision", "recall", "f1",
            "cross_val_score", "GridSearchCV",
            "QuantumCircuit", "execute", "transpile",
        ],
        "package_keywords": [
            "sklearn", "scikit-learn",
            "tensorflow", "keras",
            "torch", "pytorch",
            "xgboost", "lightgbm", "catboost",
            "qiskit", "pennylane", "cirq",
        ],
    },
    "text_processing": {
        "call_keywords": [
            "split", "join", "replace", "strip", "lstrip", "rstrip",
            "lower", "upper", "capitalize", "title",
            "startswith", "endswith", "find", "index",
            "match", "search", "findall", "sub", "compile",
            "encode", "decode", "format",
            "tokenize", "lemmatize", "stem",
            "parse", "extract",
        ],
        "package_keywords": [
            "re", "string", "textwrap",
            "nltk", "spacy", "transformers",
            "beautifulsoup", "bs4", "lxml",
            "html", "xml", "markdown",
        ],
    },
    "system_ops": {
        "call_keywords": [
            "mkdir", "makedirs", "rmdir", "remove", "unlink",
            "listdir", "scandir", "walk", "glob",
            "getcwd", "chdir", "rename", "copy",
            "environ", "getenv", "setenv",
            "argv", "exit", "getpid",
            "subprocess", "Popen", "call", "check_output",
            "Thread", "Process", "Lock", "Queue",
            "signal", "alarm",
        ],
        "package_keywords": [
            "os", "sys", "subprocess", "pathlib",
            "shutil", "glob", "fnmatch",
            "threading", "multiprocessing",
            "signal", "atexit", "tempfile",
        ],
    },
}


# =============================================================================
# Pre-compiled regex patterns for efficient matching
# =============================================================================

def _compile_patterns() -> Dict[str, Dict[str, re.Pattern]]:
    """Compile keyword lists into regex patterns for fast matching."""
    compiled = {}
    for category, keywords in TAXONOMY.items():
        # For call keywords: match as word boundaries to avoid false positives
        call_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in keywords["call_keywords"]) + r')\b',
            re.IGNORECASE,
        )
        # For package keywords: match in import statements
        pkg_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in keywords["package_keywords"]) + r')\b',
            re.IGNORECASE,
        )
        compiled[category] = {
            "call_pattern": call_pattern,
            "pkg_pattern": pkg_pattern,
        }
    return compiled


COMPILED_PATTERNS = _compile_patterns()


def match_category(code_text: str, category: str, pattern_type: str = "call") -> bool:
    """
    Check if code text matches a behavioral category.

    Args:
        code_text: The code snippet or original code text
        category: One of BEHAVIORAL_CATEGORIES
        pattern_type: "call" for code_snippet matching, "pkg" for import matching

    Returns:
        True if the text matches the category pattern
    """
    if not code_text:
        return False
    key = f"{pattern_type}_pattern"
    pattern = COMPILED_PATTERNS[category].get(key)
    if pattern is None:
        return False
    return bool(pattern.search(code_text))
