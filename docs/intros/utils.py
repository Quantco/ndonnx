# Copyright (c) QuantCo 2024-2024
# SPDX-License-Identifier: LicenseRef-QuantCo

import pandas as pd

FEATURES = [
    "VehBrand",
    "VehGas",
    "Region",
    "Area",
    "DrivAge",
    "VehAge",
    "VehPower",
    "BonusMalus",
    "Density",
]


def load_test_data():
    df = pd.read_parquet("input.parquet")

    out = {}

    for k in FEATURES:
        out[k] = df[k].to_numpy()

    return out
