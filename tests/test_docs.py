# Copyright (c) QuantCo 2023-2025
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from pytest_examples import CodeExample, EvalExample, find_examples


@pytest.mark.parametrize("example", find_examples("docs/"), ids=str)
def test_docs_examples(example: CodeExample, eval_example: EvalExample):
    if "docs/modelconversion.md" in str(example.path):
        pytest.skip(reason="interdependent snippets")
    eval_example.run(example)
