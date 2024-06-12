# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass


@dataclass
class Schema:
    type_name: str
    author: str
    meta: object = None
