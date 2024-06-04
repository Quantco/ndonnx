# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

from dataclasses import dataclass


@dataclass
class Schema:
    type_name: str
    author: str
    meta: object = None
