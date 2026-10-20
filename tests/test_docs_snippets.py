# Copyright (c) QuantCo 2026-2026
# SPDX-License-Identifier: BSD-3-Clause

import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parent.parent / "docs"

_PYTHON_FENCE = re.compile(r"^```python\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def _python_blocks(path: Path) -> list[tuple[int, str]]:
    """Return the zero-based starting line and source of each Python fence block."""
    text = path.read_text()
    return [
        (text.count("\n", 0, match.start(1)), match.group(1))
        for match in _PYTHON_FENCE.finditer(text)
    ]


@pytest.mark.parametrize(
    "path",
    [path for path in sorted(DOCS_DIR.rglob("*.md")) if _python_blocks(path)],
    ids=lambda path: str(path.relative_to(DOCS_DIR)),
)
def test_docs_snippets(path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Snippets may write files (e.g. ONNX models) to the working directory.
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "__main__"}
    for lineno, source in _python_blocks(path):
        # Leading newlines align line numbers in tracebacks with the Markdown file.
        code = compile("\n" * lineno + source, str(path), "exec")
        exec(code, namespace)

    # Snippets may demonstrate tests; run them, too.
    for name, obj in list(namespace.items()):
        if name.startswith("test_") and callable(obj):
            obj()
