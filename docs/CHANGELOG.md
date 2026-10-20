# Changelog

## 0.23.1 (unreleased)

**Other change**

- `ndonnx.cumulative_sum` with `include_initial=True` now yields more efficient graphs.
- Ndonnx produces more efficient graphs for `where` operations with boolean operands.

## 0.23.0 (2026-09-29)

**Breaking change**

- Removed functions, methods, and aliases that have been deprecated since 0.10.0. Use the following replacements instead:
    - `ndonnx.Array.null` → [`ndonnx.extensions.get_mask()`][ndonnx.extensions.get_mask]
    - `ndonnx.Array.values` → [`ndonnx.extensions.get_data()`][ndonnx.extensions.get_data]
    - `ndonnx.Array.to_numpy` → [`ndonnx.Array.unwrap_numpy()`][ndonnx.Array.unwrap_numpy]
    - `ndonnx.Array.spox_var` → [`ndonnx.Array.unwrap_spox()`][ndonnx.Array.unwrap_spox] (or [`ndonnx.Array.disassemble()`][ndonnx.Array.disassemble])
    - `ndonnx.DType.to_numpy_dtype` → [`ndonnx.DType.unwrap_numpy()`][ndonnx.DType.unwrap_numpy]
    - `ndonnx.shape` → [`ndonnx.Array.dynamic_shape`][ndonnx.Array.dynamic_shape]
    - `ndonnx.additional` → [`ndonnx.extensions`][ndonnx.extensions]
    - `ndonnx.array` → [`ndonnx.argument()`][ndonnx.argument]
    - `ndonnx.from_spox_var` → [`ndonnx.asarray()`][ndonnx.asarray]
    - `ndonnx.Floating` → [`ndonnx.extensions.is_float_dtype()`][ndonnx.extensions.is_float_dtype]
    - `ndonnx.Integral` → [`ndonnx.extensions.is_integer_dtype()`][ndonnx.extensions.is_integer_dtype]
    - `ndonnx.Numerical` → [`ndonnx.extensions.is_numeric_dtype()`][ndonnx.extensions.is_numeric_dtype]
    - `ndonnx.CoreType` → [`ndonnx.extensions.is_onnx_dtype()`][ndonnx.extensions.is_onnx_dtype]
    - `ndonnx.Nullable` and `ndonnx.NullableCore` → [`ndonnx.extensions.is_nullable_dtype()`][ndonnx.extensions.is_nullable_dtype]
    - `ndonnx.NullableFloating` → [`ndonnx.extensions.is_nullable_float_dtype()`][ndonnx.extensions.is_nullable_float_dtype]
    - `ndonnx.NullableIntegral` → [`ndonnx.extensions.is_nullable_integer_dtype()`][ndonnx.extensions.is_nullable_integer_dtype]
    - `ndonnx.UnsupportedOperationError` and `ndonnx.CastError` → these are no longer raised; catch `TypeError` instead.

**New feature**

- ndonnx's public interface now accepts objects of the following NumPy types (or a subset thereof):
    - `numpy.float16`
    - `numpy.float32`
    - `numpy.float64`
    - `numpy.int8`
    - `numpy.int16`
    - `numpy.int32`
    - `numpy.int64`
    - `numpy.uint8`
    - `numpy.uint16`
    - `numpy.uint32`
    - `numpy.uint64`
    - `numpy.bool`
    - `numpy.datetime64`
    - `numpy.timedelta64`

**Other change**

- Models built with ndonnx now expect the operator support found in `onnxruntime >=1.26.0`.

## 0.22.0 (2026-09-21)

**New features**

- Added support for [`ndonnx.log1p()`][ndonnx.log1p] and [`ndonnx.expm1()`][ndonnx.expm1].
- Most functions and methods accepting `dtype` as an argument now support the following additional aliases: `datetime64[s]`, `datetime64[ms]`, `datetime64[us]`, `datetime64[ns]`, `timedelta64[s]`, `timedelta64[ms]`, `timedelta64[us]`, `timedelta64[ns]`.
- Arrays with timedelta or datetime data types now support the `ndonnx.extensions.isin` operation.

**Other change**

- Fixed deprecation warnings emitted from `numpy >=2.5`.

## 0.21.0 (2026-08-07)

**Dependencies**

- ndonnx now requires `python>=3.11` and `numpy>=2`.

**New feature**

- Most functions and methods accepting `dtype` as an argument now support the following aliases:
    - `bool`: `ndonnx.bool`
    - `int`: `ndonnx.int64`
    - `float`: `ndonnx.float64`
    - `"bool"`: `ndonnx.bool`
    - `"int"`: `ndonnx.int64`
    - `"float"`: `ndonnx.float64`
    - `"int8"`: `ndonnx.int8`
    - `"int16"`: `ndonnx.int16`
    - `"int32"`: `ndonnx.int32`
    - `"int64"`: `ndonnx.int64`
    - `"uint8"`: `ndonnx.uint8`
    - `"uint16"`: `ndonnx.uint16`
    - `"uint32"`: `ndonnx.uint32`
    - `"uint64"`: `ndonnx.uint64`
    - `"float16"`: `ndonnx.float16`
    - `"float32"`: `ndonnx.float32`
    - `"float64"`: `ndonnx.float64`

## 0.20.0 (2026-06-16)

**New feature**

- The `:func:ndonnx.build` function gained the optional `input_prefix` and `output_prefix` arguments and the `ndonnx_schema` metadata schema was amended accordingly.

## 0.19.0 (2026-05-15)

**New feature**

- The [`extensions`][ndonnx.extensions] module now publicly exposes the [`is_nullable_numeric_dtype()`][ndonnx.extensions.is_nullable_numeric_dtype] and [`get_data()`][ndonnx.extensions.get_data] functions.

## 0.18.2 (2026-05-06)

**Bug fix**

- Fixed an issue whereby any unused inputs dropped via the `drop_unused` argument to [`ndonnx.build()`][ndonnx.build] were still included in `ndonnx_schema`.

## 0.18.1 (2026-04-22)

**Bug fix**

- Fixed a regression that forbade using the modulo operator between integers and floats.

## 0.18.0 (2026-04-07)

**New feature**

- [`ndonnx.take()`][ndonnx.take] and [`ndonnx.take_along_axis()`][ndonnx.take_along_axis] can now be used with datetime and timedelta data types.

**Bug fix**

- [`ndonnx.take()`][ndonnx.take] and [`ndonnx.take_along_axis()`][ndonnx.take_along_axis] now validate their `indices` and `axis` arguments in adherence with the array-api standard.

## 0.17.4 (2026-03-23)

**Bug fixes**

- Reverse true division now works for nullable numeric arrays, so expressions such as `1 / arr` behave correctly.
- [`ndonnx.result_type()`][ndonnx.result_type] now raises `ValueError` if called without any arguments.
- Invalid array-based slice bounds now raise `IndexError` during indexing.
- Datetime `put` operations now reject values with a mismatched dtype.
- Boolean mask `__setitem__` now raises `TypeError` when the value has too many dimensions, matching NumPy.

## 0.17.3 (2026-03-10)

**Bug fix**

- Integer floordiv (`//`) is now computed correctly even for very large integers.

## 0.17.2 (2026-02-13)

**Other change**

- [`ndonnx.Array.dynamic_size()`][ndonnx.Array.dynamic_size] now produces more efficient graphs.

## 0.17.1 (2025-12-05)

**Bug fix**

- The masks of nullable arrays are now concatenated correctly.

## 0.17.0 (2025-12-01)

**Breaking change**

- [`ndonnx.clip()`][ndonnx.clip] now behaves like NumPy when an integer dtype array and float type min or max values are given.

**Other change**

- Reduction operations now produce more efficient graphs.

## 0.16.0 (2025-08-20)

**Bug fix**

- [`ndonnx.repeat()`][ndonnx.repeat] now works correctly for zero-sized inputs on the minimum supported onnxruntime 1.20.1.

**New feature**

- [`ndonnx.max()`][ndonnx.max] ([`ndonnx.min()`][ndonnx.min]) now returns the minimum (maximum) value for the input data type if the reduction takes place over a zero-sized input.
- [`ndonnx.arange()`][ndonnx.arange] now also allows arrays as inputs.

## 0.15.0 (2025-08-13)

**New feature**

- [`ndonnx.repeat()`][ndonnx.repeat] is now implemented for all built-in data types.

**Other changes**

- Improve stacklevel presented of various deprecation warnings.

## 0.14.0 (2025-07-21)

**Bug fixes**

- Fix a bug in `ndonnx.Array.__setitem__()` that occurred when all of the following applied:
    - An `Ellipsis` was part of the key
    - The `Ellipsis` expanded to at least one dimension
    - The `Ellipsis` was not the last element of the key
    - The assigned value was not a scalar or 1D array with length 1.
- [`ndonnx.Array.dynamic_shape()`][ndonnx.Array.dynamic_shape] now returns a rank-0 array for all input ranks.
- The error message of the `IndexingError` raise by `ndonnx.Array.__setitem__()` when providing a tuple-key containing int64-arrays is now accurate.
- Using `slice` objects in the `ndonnx.Array.__setitem__()` no longer require value propagation.
- `ndonnx.Array.__setitem__()` now correctly handles boolean masks for arrays of two or more dimensions.
- Operations between NumPy and ndonnx arrays now correctly call the reverse dunder methods such as `ndonnx.Array.__radd__()` where appropriate.
- Operations between `numpy.generic` and [`ndonnx.Array`][ndonnx.Array] objects now follow the regular type promotion logic.
- The following dunder operations on [`ndonnx.Array`][ndonnx.Array] now correctly return `NotImplemented` if one of the operands is not of type `numpy.ndarray | ndonnx.Array | bool | str | int | float`: `__add__`, `__radd__`, `__and__`, `__rand__`, `__floordiv__`, `__rfloordiv__`, `__ge__`, `__gt__`, `__le__`, `__lshift__`, `__rlshift__`, `__lt__`, `__matmul__`, `__rmatmul__`, `__mod__`, `__rmod__`, `__mul__`, `__rmul__`, `__or__`, `__ror__`, `__pow__`, `__rpow__`, `__rshift__`, `__rrshift__`, `__sub__`, `__rsub__`, `__truediv__`, `__rtruediv__`, `__xor__`, `__rxor__`, `__eq__`, `__ne__`

**New workarounds for missing onnxruntime implementations**

- [`ndx.any()`][ndonnx.any] and [`ndx.all()`][ndonnx.all] now correctly propagate values even if the input is zero-sized.
- [`ndonnx.arange()`][ndonnx.arange] now follows NumPy's semantics for extremely large start, stop, and step values.
- [`ndonnx.min()`][ndonnx.min], [`ndonnx.max()`][ndonnx.max], [`ndonnx.minimum()`][ndonnx.minimum], [`ndonnx.maximum()`][ndonnx.maximum], and [`ndonnx.clip()`][ndonnx.clip] now produce correct results for very large values in int64 arrays.

## 0.13.0 (2025-05-27)

**Bug fixes**

- [`ndonnx.concat()`][ndonnx.concat] no longer raises an error if `axis=None`, the resulting data type is `int32` or `int64`, and one of the provided arrays is zero-sized.
- `ndonnx.__array_namespace_info__.capabilities()()` now reports the number of supported dimensions via the `"max dimensions"` entry rather than `"max rank"`.
- Add missing onnxruntime workaround for `uint32` inputs to [`ndonnx.min()`][ndonnx.min] and [`ndonnx.max()`][ndonnx.max].
- Fix array instantiation with [`ndonnx.asarray()`][ndonnx.asarray] and very large Python integers for `uint64` data types.
- Fix passing an Python scalar as the second argument to [`ndonnx.where()`][ndonnx.where].
- Calling [`ndonnx.roll()`][ndonnx.roll] on zero-sized inputs no longer causes a segfault on Linux.

**New features**

- The [`ndonnx.TimeDelta64DType`][ndonnx.TimeDelta64DType] and [`ndonnx.DateTime64DType`][ndonnx.DateTime64DType] gained support for milli and microseconds as units.
- [`ndonnx.where()`][ndonnx.where] now promotes time units between the two branches.
- Addition, multiplication, division, and subtraction between arrays with timedelta or datetime data types now support promotion between time units.
- Comparison operations between arrays with timedelta or datetime data types now support promotion between time units.
- Added `ndonnx.__array_api_version__` reporting the latest supported version of the Array API specification.

## 0.12.0 (2025-05-15)

**Bug fix**

- The modulo operator (`%`) now correctly follows Python's semantics if the second argument is negative.

**New features**

- Support the 2024.12 version of the Array API (except `nextafter`).

## 0.11.0 (2025-05-08)

**Bug fix**

- [`ndonnx.mean()`][ndonnx.mean], [`ndonnx.var()`][ndonnx.var], and [`ndonnx.std()`][ndonnx.std] now produce correct results if axes outside the range of `[-1, 1]` are given.

**Other change**

- Address various deprecation warnings.

## 0.10.1 (2025-04-01)

Technical release to address a PyPI publishing issue.

## 0.10.0 (2025-04-01)

**Breaking change**

- Removed the deprecated `ndonnx.promote_nullable()` function. Use [`ndonnx.extensions.make_nullable()`][ndonnx.extensions.make_nullable] instead.
- The existing (private) tooling to define custom data types has been removed.
- The `ndonnx.Array.len()` member function has been removed since it is neither defined by `numpy.ndarray` nor the Array-API.
- [`ndonnx.Array.size()`][ndonnx.Array.size] now returns `int | None` in accordance to with the Array-API rather than a [`Array`][ndonnx.Array] instance.

**Bug fixes**

- The following functions now correctly take the `keepdims` argument into account:
    - [`ndonnx.Array.sum()`][ndonnx.Array.sum]
    - [`ndonnx.Array.prod()`][ndonnx.Array.prod]
    - [`ndonnx.Array.max()`][ndonnx.Array.max]
    - [`ndonnx.Array.min()`][ndonnx.Array.min]
    - [`ndonnx.Array.all()`][ndonnx.Array.all]
    - [`ndonnx.Array.any()`][ndonnx.Array.any]

**New features**

- ndonnx gained partial support for `float16` data types.
- The following functions are now exposed in the ndonnx namespace in accordance to the Array-API standard:
    - `ndonnx.__array_namespace_info__()`
    - [`ndonnx.maximum()`][ndonnx.maximum]
    - [`ndonnx.minimum()`][ndonnx.minimum]
    - [`ndonnx.meshgrid()`][ndonnx.meshgrid]
    - [`ndonnx.moveaxis()`][ndonnx.moveaxis]
    - [`ndonnx.tile()`][ndonnx.tile]
    - [`ndonnx.unstack()`][ndonnx.unstack]
    - [`ndonnx.vecdot()`][ndonnx.vecdot]
- The newly added [`ndonnx.extensions`][ndonnx.extensions] module exposes the following functions:
    - [`datetime_to_year_month_day()`][ndonnx.extensions.datetime_to_year_month_day]
    - [`fill_null()`][ndonnx.extensions.fill_null]
    - [`get_mask()`][ndonnx.extensions.get_mask]
    - [`is_float_dtype()`][ndonnx.extensions.is_float_dtype]
    - [`is_integer_dtype()`][ndonnx.extensions.is_integer_dtype]
    - [`is_nullable_dtype()`][ndonnx.extensions.is_nullable_dtype]
    - [`is_nullable_float_dtype()`][ndonnx.extensions.is_nullable_float_dtype]
    - [`is_nullable_integer_dtype()`][ndonnx.extensions.is_nullable_integer_dtype]
    - [`is_numeric_dtype()`][ndonnx.extensions.is_numeric_dtype]
    - [`is_onnx_dtype()`][ndonnx.extensions.is_onnx_dtype]
    - [`is_signed_integer_dtype()`][ndonnx.extensions.is_signed_integer_dtype]
    - `s_time_unit()`
    - [`is_unsigned_integer_dtype()`][ndonnx.extensions.is_unsigned_integer_dtype]
    - [`isin()`][ndonnx.extensions.isin]
    - [`make_nullable()`][ndonnx.extensions.make_nullable]
    - [`put()`][ndonnx.extensions.put]
    - `shape()` (deprecated in favor of [`ndonnx.Array.dynamic_shape()`][ndonnx.Array.dynamic_shape])
    - [`static_map()`][ndonnx.extensions.static_map]
- Arrays now expose the [`ndonnx.Array.device()`][ndonnx.Array.device] property to improve Array API compatibility. Note that serializing an ONNX model inherently postpones device placement decisions to the runtime so currently one abstract device is supported.
- The [`Array`][ndonnx.Array] object gained the following member functions:
    - [`disassemble()`][ndonnx.Array.disassemble]
    - [`dynamic_shape()`][ndonnx.Array.dynamic_shape]
    - [`dynamic_size()`][ndonnx.Array.dynamic_size]
    - [`unwrap_numpy()`][ndonnx.Array.unwrap_numpy]
    - [`unwrap_spox()`][ndonnx.Array.unwrap_spox]

**Deprecations**

- `ndonnx.array()` is deprecated in favor of [`ndonnx.argument()`][ndonnx.argument].
- `ndonnx.additional` is deprecated in favor of [`ndonnx.extensions()`][ndonnx.extensions].
- `ndonnx.from_spox_var()` is deprecated in favor of [`ndonnx.asarray()`][ndonnx.asarray].
- `ndonnx.Nullable` is deprecated in favor of [`ndonnx.extensions.is_nullable_dtype()`][ndonnx.extensions.is_nullable_dtype]
- `ndonnx.NullableFloating` is deprecated in favor of [`ndonnx.extensions.is_nullable_float_dtype()`][ndonnx.extensions.is_nullable_float_dtype]
- `ndonnx.NullableIntegral` is deprecated in favor of [`ndonnx.extensions.is_nullable_integer_dtype()`][ndonnx.extensions.is_nullable_integer_dtype]
- `ndonnx.Floating` is deprecated in favor of [`ndonnx.extensions.is_float_dtype()`][ndonnx.extensions.is_float_dtype]
- `ndonnx.Integral` is deprecated in favor of [`ndonnx.extensions.is_integer_dtype()`][ndonnx.extensions.is_integer_dtype]
- `ndonnx.Numerical` is deprecated in favor of [`ndonnx.extensions.is_numeric_dtype()`][ndonnx.extensions.is_numeric_dtype]
- `ndonnx.CoreType` is deprecated in favor of [`ndonnx.extensions.is_onnx_dtype()`][ndonnx.extensions.is_onnx_dtype]
- `ndonnx.NullableCore` is deprecated in favor of [`ndonnx.extensions.is_nullable_dtype()`][ndonnx.extensions.is_nullable_dtype]
- `ndonnx.UnsupportedOperationError` is deprecated in favor of [`TypeError`][TypeError]
- `ndonnx.CastError` is deprecated in favor of [`TypeError`][TypeError]

**Other changes**

- `make_nullable()` can no longer be used for custom data types.

## 0.9.3 (2024-10-25)

- Reduced the number of unnecessary casts in [`ndonnx.argmax()`][ndonnx.argmax] and [`ndonnx.argmin()`][ndonnx.argmin].

## 0.9.2 (2024-10-03)

- Technical release with source distribution.

## 0.9.1 (2024-10-01)

**Bug fix**

- Fixed a bug in the construction of nullable arrays using [`ndonnx.asarray()`][ndonnx.asarray] where the shape of the null field would not match the values field if the provided `np.ma.MaskedArray`'s mask was scalar.
- Fixed a bug in the implementation of [`ndonnx.ones_like()`][ndonnx.ones_like] where the static shape was being used to construct the array of ones.

## 0.9.0 (2024-08-30)

**New features**

- User defined data types can now define how arrays with that dtype are constructed by implementing the `make_array` function.
- User defined data types can now define how they are indexed (via `__getitem__`) by implementing the `getitem` function.
- `ndonnx.NullableCore` is now public, encapsulating nullable variants of \`CoreType\`s exported by ndonnx.

**Bug fixes**

- Various operations that depend on the array's shape have been updated to work correctly with lazy arrays.
- [`ndonnx.cumulative_sum()`][ndonnx.cumulative_sum] now correctly applies the `include_initial` parameter and works around missing onnxruntime kernels for unsigned integral types.
- `ndonnx.additional.make_nullable()` applies broadcasting to the provided null array (instead of reshape like it did previously). This allows writing `make_nullable(x, False)` to turn an array into nullable.
- User-defined data types that implement `ndonnx._core.UniformShapeOperations` may now implement [`ndonnx.where()`][ndonnx.where] without requiring both data types be promotable.

**Breaking change**

- Iterating over dynamic dimensions of [`Array`][ndonnx.Array] is no longer allowed since it commonly lead to infinite loops when used without an explicit break condition.

## 0.8.0 (2024-08-22)

**Bug fixes**

- Fixes parsing numpy arrays of type `object` (consisting of strings) as `utf8`. Previously this worked correctly only for 1d arrays.

**Breaking change**

- [`ndonnx.Array.shape()`][ndonnx.Array.shape] now strictly returns a `tuple[int | None, ...]`, with unknown dimensions denoted by `None`. This relies on ONNX shape inference for lazy arrays.

## 0.7.0 (2024-08-12)

**New features**

- Expose the [`ndonnx.isdtype()`][ndonnx.isdtype] function.

- Custom data types can now override array functions:
    - [`ndonnx.zeros()`][ndonnx.zeros]
    - [`ndonnx.zeros_like()`][ndonnx.zeros_like]
    - [`ndonnx.ones()`][ndonnx.ones]
    - [`ndonnx.ones_like()`][ndonnx.ones_like]
    - [`ndonnx.full()`][ndonnx.full]
    - [`ndonnx.full_like()`][ndonnx.full_like]
    - [`ndonnx.arange()`][ndonnx.arange]
    - [`ndonnx.arange()`][ndonnx.arange]
    - [`ndonnx.eye()`][ndonnx.eye]
    - [`ndonnx.tril()`][ndonnx.tril]
    - [`ndonnx.triu()`][ndonnx.triu]
    - [`ndonnx.linspace()`][ndonnx.linspace]
    - [`ndonnx.where()`][ndonnx.where]

- The `ndonnx._experimental.UniformShapeOperations` now provides implementations of shape operations that are generic across all data types where each constituent field has the same shape (that of the overall array).

**Other changes**

- Fixed various deprecation warnings.
- Invoking a function using arrays with data types that lack a corresponding implementation now raise a `UnsupportedOperationError`.

**Bug fixes**

- Numerical operations like [`sin()`][ndonnx.sin] now raise `UnsupportedOperationError` when invoked using invalid data types like `ndx.utf8` rather than implicitly casting.
- Fixes bug causing a promotion error when implementing numerical operations like [`add()`][ndonnx.add] that involve type promotion.
- Fixes scalar promotion logic to more accurately reflect the Array API standard. Promotion requires at least one array to be present and scalars adopt the dtype of the arrays being promoted with it. `ndx.utf8` and `ndx.nutf8` cannot be promoted with any other dtypes.
- Fixes failure when broadcasting nullable data type arrays together in [`broadcast_arrays()`][ndonnx.broadcast_arrays].

## 0.6.1 (2024-07-12)

**Bug fixes**

- Division now complies more strictly with the Array API standard by returning a floating-point result regardless of input data types.

## 0.6.0 (2024-07-11)

**Other changes**

- `ndonnx.promote_nullable` is now publicly exported.

## 0.5.0 (2024-07-01)

**Other changes**

- ndonnx now exports type annotations.

**Bug fixes**

- `__array_namespace__` now accepts the optional `api_version` argument to specify the version of the Array API to use.

## 0.4.0 (2024-05-16)

**Breaking changes**

- The constant propagated value is no longer accessed from the `eager_value` property but instead the `to_numpy()` method.
- Non Array API functions have been moved to the `ndonnx.additional` namespace.
