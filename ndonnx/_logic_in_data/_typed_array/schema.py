# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

# class SchemaInfo:
#     defining_library: str
#     version: int
#     dtype: str
#     components: dict[str, Literal["int8", "bool", "str"]]

# class BaseNumPySchema(ABC):
#     @abstractmethod
#     def disassemble(self, np_arr: np.ndarray) -> dict[str, np.ndarray]:
#         ...

#     @abstractmethod
#     def assemble(self, np_arrs: dict[str, np.ndarray]) -> np.ndarray:
#         ...

# class NumPySchema(BaseNumPySchema):
#     def __init__(self, schema_info: SchemaInfo):
#         if schema_info.defining_library != "ndonnx":
#             raise ValueError(
#                 f"expected ndonnx schema; found `{schema_info.defining_library}`"
#             )
#         if schema_info.version != 1:
#             raise ValueError(
#                 f"unsupported schema version `{schema_info.version}`"
#             )
#         dtype_map = {
#             "int8": dtypes.int8,
#         }
#         self.dtype = dtype_map[schema_info.dtype]
#         self.version = schema_info.version
#         ...

#     def disassemble(self, np_arr: np.ndarray) -> dict[str, np.ndarray]:
#         if np.ma.is_masked(np_arr):
#             raise TypeError("expected non-masked input")

#         expected_np_dtype = as_numpy(self.dtype)
#         if np_arr.dtype != expected_np_dtype:
#             raise TypeError(
#                 f"expected `{expected_np_dtype}` data type; found `{np_arr.dtype}`"
#             )
#         if self.version == 1:
#             return {"var": np_arr}
#         # This was checked in __init__
#         raise NotImplementedError("Unsupported version")
