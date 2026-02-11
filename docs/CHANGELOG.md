# Changelog

## 0.17.1 (2025-12-05)

**Bug fix**

- The masks of nullable arrays are now concatenated correctly.

## 0.17.0 (2025-12-01)

**Breaking change**

- `ndonnx.clip` now behaves like NumPy when an integer dtype array and
  float type min or max values are given.

**Other change**

- Reduction operations now produce more efficient graphs.

## 0.16.0 (2025-08-20)

**Bug fix**

- `ndonnx.repeat` now works correctly for zero-sized inputs on the
  minimum supported onnxruntime 1.20.1.

**New feature**

- `ndonnx.max` (`ndonnx.min`) now returns the minimum (maximum) value
  for the input data type if the reduction takes place over a zero-sized
  input.
- `ndonnx.arange` now also allows arrays as inputs.

## 0.15.0 (2025-08-13)

**New feature**

- `ndonnx.repeat` is now implemented for all built-in data types.

**Other changes**

- Improve stacklevel presented of various deprecation warnings.

## 0.14.0 (2025-07-21)

**Bug fixes**

- Fix a bug in `ndonnx.Array.__setitem__` that occurred when all of the
  following applied:
  - An `Ellipsis` was part of the key
  - The `Ellipsis` expanded to at least one dimension
  - The `Ellipsis` was not the last element of the key
  - The assigned value was not a scalar or 1D array with length 1.
- `ndonnx.Array.dynamic_shape` now returns a rank-0 array for all input
  ranks.
- The error message of the `IndexingError` raise by
  `ndonnx.Array.__setitem__` when providing a tuple-key containing
  int64-arrays is now accurate.
- Using <span class="title-ref">slice</span> objects in the
  `ndonnx.Array.__setitem__` no longer require value propagation.
- `ndonnx.Array.__setitem__` now correctly handles boolean masks for
  arrays of two or more dimensions.
- Operations between NumPy and ndonnx arrays now correctly call the
  reverse dunder methods such as `ndonnx.Array.__radd__` where
  appropriate.
- Operations between `numpy.generic` and `ndonnx.Array` objects now
  follow the regular type promotion logic.
- The following dunder operations on `ndonnx.Array` now correctly return
  `NotImplemented` if one of the operands is not of type
  `numpy.ndarray | ndonnx.Array | bool | str | int | float`: `__add__`,
  `__radd__`, `__and__`, `__rand__`, `__floordiv__`, `__rfloordiv__`,
  `__ge__`, `__gt__`, `__le__`, `__lshift__`, `__rlshift__`, `__lt__`,
  `__matmul__`, `__rmatmul__`, `__mod__`, `__rmod__`, `__mul__`,
  `__rmul__`, `__or__`, `__ror__`, `__pow__`, `__rpow__`, `__rshift__`,
  `__rrshift__`, `__sub__`, `__rsub__`, `__truediv__`, `__rtruediv__`,
  `__xor__`, `__rxor__`, `__eq__`, `__ne__`

**New workarounds for missing onnxruntime implementations**

- `ndx.any` and `ndx.all` now correctly propagate values even if the
  input is zero-sized.
- `ndonnx.arange` now follows NumPy's semantics for extremely large
  start, stop, and step values.
- `ndonnx.min`, `ndonnx.max`, `ndonnx.minimum`, `ndonnx.maximum`, and
  `ndonnx.clip` now produce correct results for very large values in
  int64 arrays.

## 0.13.0 (2025-05-27)

**Bug fixes**

- `ndonnx.concat` no longer raises an error if `axis=None`, the
  resulting data type is `int32` or `int64`, and one of the provided
  arrays is zero-sized.
- `ndonnx.__array_namespace_info__.capabilities()` now reports the
  number of supported dimensions via the `"max dimensions"` entry rather
  than `"max rank"`.
- Add missing onnxruntime workaround for `uint32` inputs to `ndonnx.min`
  and `ndonnx.max`.
- Fix array instantiation with `ndonnx.asarray` and very large Python
  integers for `uint64` data types.
- Fix passing an Python scalar as the second argument to `ndonnx.where`.
- Calling `ndonnx.roll` on zero-sized inputs no longer causes a segfault
  on Linux.

**New features**

- The `ndonnx.TimeDelta64DType` and `ndonnx.DateTime64DType` gained
  support for milli and microseconds as units.
- `ndonnx.where` now promotes time units between the two branches.
- Addition, multiplication, division, and subtraction between arrays
  with timedelta or datetime data types now support promotion between
  time units.
- Comparison operations between arrays with timedelta or datetime data
  types now support promotion between time units.
- Added `ndonnx.__array_api_version__` reporting the latest supported
  version of the Array API specification.

## 0.12.0 (2025-05-15)

**Bug fix**

- The modulo operator (`%`) now correctly follows Python's semantics if
  the second argument is negative.

**New features**

- Support the 2024.12 version of the Array API (except `nextafter`).

## 0.11.0 (2025-05-08)

**Bug fix**

- `ndonnx.mean`, `ndonnx.var`, and `ndonnx.std` now produce correct
  results if axes outside the range of <span class="title-ref">\[-1,
  1\]</span> are given.

**Other change**

- Address various deprecation warnings.

## 0.10.1 (2025-04-01)

Technical release to address a PyPI publishing issue.

## 0.10.0 (2025-04-01)

**Breaking change**

- Removed the deprecated `ndonnx.promote_nullable` function. Use
  `ndonnx.extensions.make_nullable` instead.
- The existing (private) tooling to define custom data types has been
  removed.
- The `ndonnx.Array.len` member function has been removed since it is
  neither defined by `numpy.ndarray` nor the Array-API.
- `ndonnx.Array.size` now returns <span class="title-ref">int \|
  None</span> in accordance to with the Array-API rather than a
  `~ndonnx.Array` instance.

**Bug fixes**

- The following functions now correctly take the `keepdims` argument
  into account:
  - `ndonnx.Array.sum`
  - `ndonnx.Array.prod`
  - `ndonnx.Array.max`
  - `ndonnx.Array.min`
  - `ndonnx.Array.all`
  - `ndonnx.Array.any`

**New features**

- ndonnx gained partial support for
  <span class="title-ref">float16</span> data types.
- The following functions are now exposed in the ndonnx namespace in
  accordance to the Array-API standard:
  - `ndonnx.__array_namespace_info__`
  - `ndonnx.maximum`
  - `ndonnx.minimum`
  - `ndonnx.meshgrid`
  - `ndonnx.moveaxis`
  - `ndonnx.tile`
  - `ndonnx.unstack`
  - `ndonnx.vecdot`
- The newly added `ndonnx.extensions` module exposes the following
  functions:
  - `~ndonnx.extensions.datetime_to_year_month_day`
  - `~ndonnx.extensions.fill_null`
  - `~ndonnx.extensions.get_mask`
  - `~ndonnx.extensions.is_float_dtype`
  - `~ndonnx.extensions.is_integer_dtype`
  - `~ndonnx.extensions.is_nullable_dtype`
  - `~ndonnx.extensions.is_nullable_float_dtype`
  - `~ndonnx.extensions.is_nullable_integer_dtype`
  - `~ndonnx.extensions.is_numeric_dtype`
  - `~ndonnx.extensions.is_onnx_dtype`
  - `~ndonnx.extensions.is_signed_integer_dtype`
  - `~ndonnx.extensions.s_time_unit`
  - `~ndonnx.extensions.is_unsigned_integer_dtype`
  - `~ndonnx.extensions.isin`
  - `~ndonnx.extensions.make_nullable`
  - `~ndonnx.extensions.put`
  - `~ndonnx.extensions.shape` (deprecated in favor of
    `ndonnx.Array.dynamic_shape`)
  - `~ndonnx.extensions.static_map`
- Arrays now expose the `ndonnx.Array.device` property to improve Array
  API compatibility. Note that serializing an ONNX model inherently
  postpones device placement decisions to the runtime so currently one
  abstract device is supported.
- The `~ndonnx.Array` object gained the following member functions:
  - `~ndonnx.Array.disassemble`
  - `~ndonnx.Array.dynamic_shape`
  - `~ndonnx.Array.dynamic_size`
  - `~ndonnx.Array.unwrap_numpy`
  - `~ndonnx.Array.unwrap_spox`

**Deprecations**

- `ndonnx.array` is deprecated in favor of `ndonnx.argument`.
- `ndonnx.additional` is deprecated in favor of `ndonnx.extensions`.
- `ndonnx.from_spox_var` is deprecated in favor of `ndonnx.asarray`.
- `ndonnx.Nullable` is deprecated in favor of
  `ndonnx.extensions.is_nullable_dtype`
- `ndonnx.NullableFloating` is deprecated in favor of
  `ndonnx.extensions.is_nullable_float_dtype`
- `ndonnx.NullableIntegral` is deprecated in favor of
  `ndonnx.extensions.is_nullable_integer_dtype`
- `ndonnx.Floating` is deprecated in favor of
  `ndonnx.extensions.is_float_dtype`
- `ndonnx.Integral` is deprecated in favor of
  `ndonnx.extensions.is_integer_dtype`
- `ndonnx.Numerical` is deprecated in favor of
  `ndonnx.extensions.is_numeric_dtype`
- `ndonnx.CoreType` is deprecated in favor of
  `ndonnx.extensions.is_onnx_dtype`
- `ndonnx.NullableCore` is deprecated in favor of
  `ndonnx.extensions.is_nullable_dtype`
- `ndonnx.UnsupportedOperationError` is deprecated in favor of
  `TypeError`
- `ndonnx.CastError` is deprecated in favor of `TypeError`

**Other changes**

- `~ndonnx.additional.make_nullable` can no longer be used for custom
  data types.

## 0.9.3 (2024-10-25)

- Reduced the number of unnecessary casts in `ndonnx.argmax` and
  `ndonnx.argmin`.

## 0.9.2 (2024-10-03)

- Technical release with source distribution.

## 0.9.1 (2024-10-01)

**Bug fix**

- Fixed a bug in the construction of nullable arrays using
  `ndonnx.asarray` where the shape of the null field would not match the
  values field if the provided
  <span class="title-ref">np.ma.MaskedArray</span>'s mask was scalar.
- Fixed a bug in the implementation of `ndonnx.ones_like` where the
  static shape was being used to construct the array of ones.

## 0.9.0 (2024-08-30)

**New features**

- User defined data types can now define how arrays with that dtype are
  constructed by implementing the `make_array` function.
- User defined data types can now define how they are indexed (via
  `__getitem__`) by implementing the `getitem` function.
- `ndonnx.NullableCore` is now public, encapsulating nullable variants
  of \`CoreType\`s exported by ndonnx.

**Bug fixes**

- Various operations that depend on the array's shape have been updated
  to work correctly with lazy arrays.
- `ndonnx.cumulative_sum` now correctly applies the `include_initial`
  parameter and works around missing onnxruntime kernels for unsigned
  integral types.
- `ndonnx.additional.make_nullable` applies broadcasting to the provided
  null array (instead of reshape like it did previously). This allows
  writing `make_nullable(x, False)` to turn an array into nullable.
- User-defined data types that implement
  `ndonnx._core.UniformShapeOperations` may now implement `ndonnx.where`
  without requiring both data types be promotable.

**Breaking change**

- Iterating over dynamic dimensions of `~ndonnx.Array` is no longer
  allowed since it commonly lead to infinite loops when used without an
  explicit break condition.

## 0.8.0 (2024-08-22)

**Bug fixes**

- Fixes parsing numpy arrays of type `object` (consisting of strings) as
  `utf8`. Previously this worked correctly only for 1d arrays.

**Breaking change**

- `ndonnx.Array.shape` now strictly returns a `tuple[int | None, ...]`,
  with unknown dimensions denoted by `None`. This relies on ONNX shape
  inference for lazy arrays.

## 0.7.0 (2024-08-12)

**New features**

- Expose the `ndonnx.isdtype` function.

- Custom data types can now override array functions:
  - `ndonnx.zeros`
  - `ndonnx.zeros_like`
  - `ndonnx.ones`
  - `ndonnx.ones_like`
  - `ndonnx.full`
  - `ndonnx.full_like`
  - `ndonnx.arange`
  - `ndonnx.arange`
  - `ndonnx.eye`
  - `ndonnx.tril`
  - `ndonnx.triu`
  - `ndonnx.linspace`
  - `ndonnx.where`

- The `ndonnx._experimental.UniformShapeOperations` now provides
  implementations of shape operations that are generic across all data
  types where each constituent field has the same shape (that of the
  overall array).

**Other changes**

- Fixed various deprecation warnings.
- Invoking a function using arrays with data types that lack a
  corresponding implementation now raise a `UnsupportedOperationError`.

**Bug fixes**

- Numerical operations like `sin` now raise `UnsupportedOperationError`
  when invoked using invalid data types like `ndx.utf8` rather than
  implicitly casting.
- Fixes bug causing a promotion error when implementing numerical
  operations like `add` that involve type promotion.
- Fixes scalar promotion logic to more accurately reflect the Array API
  standard. Promotion requires at least one array to be present and
  scalars adopt the dtype of the arrays being promoted with it.
  <span class="title-ref">ndx.utf8</span> and
  <span class="title-ref">ndx.nutf8</span> cannot be promoted with any
  other dtypes.
- Fixes failure when broadcasting nullable data type arrays together in
  `broadcast_arrays`.

## 0.6.1 (2024-07-12)

**Bug fixes**

- Division now complies more strictly with the Array API standard by
  returning a floating-point result regardless of input data types.

## 0.6.0 (2024-07-11)

**Other changes**

- `ndonnx.promote_nullable` is now publicly exported.

## 0.5.0 (2024-07-01)

**Other changes**

- ndonnx now exports type annotations.

**Bug fixes**

- `__array_namespace__` now accepts the optional `api_version` argument
  to specify the version of the Array API to use.

## 0.4.0 (2024-05-16)

**Breaking changes**

- The constant propagated value is no longer accessed from the
  `eager_value` property but instead the `to_numpy()` method.
- Non Array API functions have been moved to the `ndonnx.additional`
  namespace.
