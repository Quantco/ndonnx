# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from typing import TypeVar

import numpy as np

ISIN_SCALAR = TypeVar("ISIN_SCALAR", int, float, str, np.datetime64, np.timedelta64)
MAPPING_KEY = TypeVar("MAPPING_KEY", int, float, str)
MAPPING_VALUE = TypeVar("MAPPING_VALUE", int, float, str)
