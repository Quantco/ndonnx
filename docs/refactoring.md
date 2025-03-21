# Summary

[summary]: #summary

The internals of `ndonnx.Array` are refactored to be based around a "typed array" class that carries the data type in its own type.
Instances of the well-known `DType` class become "builders" for typed arrays.
The impact on the public API is minor but existent.

# Motivation

[motivation]: #motivation

The implementation discussed in the prior art section provides very few statically verifiable guarantees and implements its own value propagation rather than relying on Spox for it.
The former point is undesirable, given the foundational nature of this library and the longevity of the created artifacts.
The latter point duplicates efforts and complicates the code base of ndonnx.
The goal of the design presented attempts to be fully array-api compliant (with the exception of complex data types), thoroughly typed, and well maintainable by utilizing common design patterns found in Python (namely linear inheritance and use of reflective dunder-methods).

# Design overview

[guide-level-explanation]: #guide-level-explanation

The public facing API continues to only expose a single `Array` class:

```python
class Array:
    """User-facing objects that makes no assumption about any data type related
    logic."""

    _tyarray: TyArrayBase

    @property
    def dtype(self) -> DType:
        return self._tyarray.dtype
```

The `DType` describes the data type of an array as is very common in Python array libraries.
The entire design presented below is constraint to the private `TyArrayBase` class - an array object which itself (largely) implements the array-api.

## Typed arrays

The abstract `TyArrayBase` class presents a base class for an array that carries its data type information in its type.
Throughout `ndonnx` we require that the data type of an object is always statically known; a constraint that is stricter than the ONNX standard itself.
Furthermore, the array-api standard prohibits mutating operations that would change the type of the underlying object (cf. `__setitem__`).
These invariances allow us to encode the data type in the class type itself.
This is beneficial since the array class also presents an obvious choice to place the implementations of various operations such as `__add__` or `reshape`, but such operations routinely require some specialization per data type.

The `TyArrayBase` describes various abstract methods that must be implemented for every type such as reshape operations, `__getitem__` and `__setitem__`.
Other operations may remain unimplemented but will raise an informative error if called such as `sum`.

Various data types my then be implemented via inheritance.
For instance, data types represented in the onnx standard are defined in `ndonnx/ndonnx/_refactor/_typed_array/onnx.py`.
All arrays with ONNX data types share the same data model:

```python
class TyArray(TyArrayBase):
    dtype: _OnnxDType
    var: Var
```

This allows us to define all type independent operations (such as `reshape`) on the `TyArray` class itself.
Operations which may not be defined for all data types or require different operators on the ONNX side are later defined in further subclasses.
For instance, `__add__` is defined once for string-arrays (`TyArrayUtf8`) using the `StringConcat` operation of the ONNX standard and once for numerical ONNX types (`TyArrayNumber`) using the `Add` operator.

Ultimately, every data type is represent with a unique array class.

While many of these operations defined on the `TyArrayBase` subclasses touch on type promotion, the discussion of that topic is deferred until after a closer examination of the `DType` class and its place in the presented design

## Array creation: `DType` as a builder instances

The `DType` class is defined as a abstract base class which is generic over the typed array that it is associated with:

```python
TY_ARRAY_BASE = TypeVar("TY_ARRAY_BASE", bound="TyArrayBase")

class DType(ABC, Generic[TY_ARRAY_BASE]):
	...
```

Since `DType` is public, all `TyArrayBase` related functionality is defined on semi-private `__ndx_*__` members.
Just as every typed array instance is associated with exactly one `DType` class, every `DType` instance is associated with exactly one typed array class.
Since each array may have an arbitrary layout, it is not possible to provide a common interface to build such arrays from the base `DType` class.
Instead, data types commonly provide there own `_build` methods which may be accessed once the type has been appropriately narrowed.

Conceptually, instances of subclasses of `DType` can be thought of as "builder" objects used to instantiate new arrays of type `TY_ARRAY_BASE`.
As such `DType` defines methods that mirror creation functions of the array-api such as `zeros`:

```python
    @abstractmethod
    def _zeros(self, shape: tuple[int, ...] | TyArrayInt64) -> TY_ARRAY_BASE: ...
```

As such, `DType` subclasses provide a place to specialize constructor functions for their respective data types.
A hierarchical inheritance may be used to avoid code duplication across similar data types (e.g. numerical ONNX data types), while the generic type annotations still provide strong guarantees during static analysis.

## Data types and type promotion

Type promotion can be thought of as an instance of array creation, too.
Compared to the previously mentioned `_zeros` method, it has the additional complication that it two data types may interact.

Before touching on the creation of new arrays after a type promotion, it is instructive to first discuss how the publci `result_type` function is implemented.
Similar to `zeros` there is a mirrored method to be found method on `DType`:

```python
class DType(ABC, Generic[TY_ARRAY_BASE]):
    ...

    @abstractmethod
    def __ndx_result_type__(self, other: DType) -> DType | NotImplementedType: ...
```

The `result_type` function is implemented following Python pattern for reflected functions (e.g. `__add__` and `__radd__`).
If the `result_type` function is presented with two `DType` instances it tries query either of them until a common `DType` is returned.
If either `DType` instance returns `NotImplemented` a error is raised.
This allows for downstream data types to define type promotion behavior with built-in data types even though the built-in ones have know knowledge of the downstream ones.

### Interaction with Python scalars

Promotion rules with Python scalars have various intricacies which prevent an implementation where Python scalars are trivially cast to rank-0 arrays of type `int64`, `float64`, `bool` or `utf8`.
Each data type may implement their own interaction with Python scalars as is best suited or dictated by the standard.

### Stateful data types

Some data types are stateful such as datetimes which carry a specific unit.
Datetime and timedelta typed-arrays and data types have been implemented as an example an can be found [here](ndonnx/ndonnx/_refactor/_typed_array/date_time.py).

## `__ndx*__` functions

Functions named in this pattern are not meant to be called directly (following the Python pattern for reflective dunder methods).
Instead, these functions are used to implement higher-level functions such as `ndonnx.result_type` which should be called instead.
`__ndx*__` may also return `NotImplemented` objects.

# Prior art

[prior-art]: #prior-art

The current data model of `ndonnx.Array` is given as:

```python
class Array:
    dtype: dtypes.StructType | dtypes.CoreType
    _fields: dict[str, Array | _CoreArray]
```

The types `dtypes.StructType` and `dtypes.CoreType` form a union of two non-inheriting data type types.
The `_CoreArray` class is defined as:

```python
class _CoreArray:
    """Thin wrapper around a `spox.Var` as well as optionally the eager propagated
    value."""

    var: Var
    _eager_value: np.ndarray | None
```

The motivation behind this design was that the `Array` object itself cannot know about how operations for a particular data type would be implemented.
`Array` was intended to treat `_fields` as an opaque blob, while the actual implementations were defined on subclasses of the data type classes.

More precisely, `CoreType` was defined as follows (with modifications for brevity):

```python
class CoreType:
	_ops: OperationsBlock

    @abstractmethod
    def _cast_from(self, array: Array) -> Array:
        """Casts the input array to this data type.

        Raise a `CastFailedError` if the cast is not possible.
        """
        ...

    @abstractmethod
    def _cast_to(self, array: Array, dtype: CoreType | StructType) -> Array:
        """Cast the input array from this data type to the specified data type `dtype`.

        Raise a `CastFailedError` if the cast is not possible.
        """
        ...

	...  # Omitted methods

class OperationsBlock:
    """Interface for data types to implement top-level functions exported by ndonnx."""

    def abs(self, x) -> ndx.Array:
        return NotImplemented
    ...

	...   # Omitted further methods
```

In practice, this design proofed difficult to maintain in practice for the various reasons listed below.

## `OperationBlock` is oblivious of the data type owning it

The `OperationBlock` was introduced to gain code reuse between related data types, but it has no notion itself which data type owns it.
This makes it difficult to implement specialization for some data types.
An example in this regard is given by `bitwise_right_shift`.
The array-api standard defines it "arithmetic" for signed integers, while the ONNX standard only defines its bit shift operators for unsigned integers.
The solution is to use floor-division for signed integers, and the specialized ONNX operator for unsigned integers.

While it would be possible to introduce a `class UnsignedOperationsImpl(OperationsBlock): ...`, this would duplicate the information already stored in the data type itself.

## Significantly reduced utility of static type checking

The signature of `Array._fields: dict` is already of very limited utility since it provides no static guarantees based on the data type.
Adjacent to this issue, `OperationsBlock` is largely untyped, too.

### Guarantees of the standard are not reflected in the type system

Since `OperationsBlock` is meant to be shared between data types, it is impossible to use it to encode type guarantees made by the standard in the signatures.
For instance, the array-api often specifies that a return type must equal the input type (e.g. for `Array.T`, but this could not be reflected in the previous design.

## Non-standard "reflected" implementation for binary operations

Python commonly uses `__r*__` to implement reflected versions of binary operators.
The previous implementation does not rely on this standard practice.
Instead, type promotion is applied in the member functions of `OperationBlock`.
At first the `OperationBlock` of the first argument is used.
If the operation fails, the `OperationBlock` of the second argument is used.

While this implementation is very similar to the Pythonic `__r*__` pattern, it is unfortunate that it does not apply the pattern directly.

Discuss prior art, both the good and the bad, in relation to this proposal.
A few examples of what this can include are:

- For language, library, cargo, tools, and compiler proposals: Does this feature exist in other programming languages and what experience have their community had?
- For community proposals: Is this done by some other community and what were their experiences with it?
- For other teams: What lessons can we learn from what other communities have done here?
- Papers: Are there any published papers or great posts that discuss this? If you have some relevant papers to refer to, this can serve as a more detailed theoretical background.

This section is intended to encourage you as an author to think about the lessons from other languages, provide readers of your RFC with a fuller picture.
If there is no prior art, that is fine - your ideas are interesting to us whether they are brand new or if it is an adaptation from other languages.

Note that while precedent set by other languages is some motivation, it does not on its own motivate an RFC.
Please also take into consideration that rust sometimes intentionally diverges from common language features.

# Unresolved questions / other remarks

[unresolved-questions]: #unresolved-questions

## Ship datetime and timedelta data types

NumPy has been shipping date time data types for a long time.
While the underlying data model leaves considerable room for debate (`float64` vs. `int64` with sentinel values vs. `int64` with a `NaT` boolean mask), the overall semantics appear much less controversial (i.e. `NaT` is always infectious).
The presented refactoring includes a rudimentary datetime and time delta implementation already.
