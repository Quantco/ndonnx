# Copyright (c) QuantCo 2023-2024
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import typing_extensions

from ndonnx._core._interface import OperationsBlock
from ndonnx._data_types.coretype import CoreType

from .schema import Schema


class StructType:
    """The ``StructType`` class defines all data types that go beyond the core data
    types present in the ONNX standard.

    Inheriting from ``StructType`` enables the implementation of more complex and domain-specific data types. The API for doing so is used to implement the nullable data types exported by ``ndonnx``. This API is not yet public but may be considered ready for experimentation.
    """

    @abstractmethod
    def _fields(self) -> dict[str, StructType | CoreType]:
        """Define the named fields of this StructType."""
        ...

    @abstractmethod
    def _parse_input(self, input: np.ndarray) -> dict:
        """This method may be used by runtime utilities. It should break down the input
        numpy array into a dictionary of values with an entry for each field in this
        StructType.

        As an example, consider defining a 128 bit data type as a StructType as follows:

            .. code-block:: python

                class My128BitType(StructType):
                    def __init__(self):
                        super().__init__(low=uint64(), high=uint64())

            Then, ``parse_input`` can be defined as follows:

            .. code-block:: python

                def _parse_input(self, input: np.ndarray) -> dict[str, np.ndarray]:
                    # We expect a numpy array of python int objects.
                    if input.dtype != object:
                        raise TypeError("Input must be an object array")
                    mask = (1 << 64) - 1
                    return {
                        "low": (input & mask).astype(np.uint64),
                        "high": (input >> 64).astype(np.uint64) & mask,
                    }
        """
        ...

    @abstractmethod
    def _assemble_output(self, fields: dict[str, np.ndarray]) -> np.ndarray:
        """This method may be used by runtime utilities, and is essentially an inverse
        of ``parse_input``. It should take a dictionary of values (representing the
        fields of the struct being defined) and compose them into a single
        representation of a value with this data type.

        As an example, consider defining a 128 bit data type as a StructType as follows:

        .. code-block:: python

            class My128BitType(StructType):
                def __init__(self):
                    super().__init__(low=uint64(), high=uint64())

        Then, ``assemble_output`` can be defined as follows:

        .. code-block:: python

            def _assemble_output(self, fields: dict[str, np.ndarray]) -> np.ndarray:
                return (fields["high"].astype(object) << 64) | fields["low"].astype(object)
        """

    @abstractmethod
    def _schema(self) -> Schema:
        """Exports a schema that describes any basic information required by runtime
        utilities to consume input values corresponding to this array."""
        ...

    @abstractmethod
    def copy(self) -> typing_extensions.Self:
        """Clone this data type."""
        ...

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) or type(self) is other

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return type(self).__name__

    def __hash__(self) -> int:
        return hash(type(self))

    _ops: OperationsBlock
    """Defines functions that may be executed using a ``ndx.Array`` of this data
    type."""
