.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

0.15.0 (2025-08-13)
-------------------

**New feature**

- :func:`ndonnx.repeat` is now implemented for all built-in data types.

**Other changes**

- Improve stacklevel presented of various deprecation warnings.


0.14.0 (2025-07-21)
-------------------

**Bug fixes**

- Fix a bug in :meth:`ndonnx.Array.__setitem__` that occurred when all of the following applied:
  - An ``Ellipsis`` was part of the key
  - The ``Ellipsis`` expanded to at least one dimension
  - The ``Ellipsis`` was not the last element of the key
  - The assigned value was not a scalar or 1D array with length 1.
- :meth:`ndonnx.Array.dynamic_shape` now returns a rank-0 array for all input ranks.
- The error message of the ``IndexingError`` raise by :meth:`ndonnx.Array.__setitem__` when providing a tuple-key containing int64-arrays is now accurate.
- Using `slice` objects in the :meth:`ndonnx.Array.__setitem__` no longer require value propagation.
- :meth:`ndonnx.Array.__setitem__` now correctly handles boolean masks for arrays of two or more dimensions.
- Operations between NumPy and ndonnx arrays now correctly call the reverse dunder methods such as :meth:`ndonnx.Array.__radd__` where appropriate.
- Operations between ``numpy.generic`` and :class:`ndonnx.Array` objects now follow the regular type promotion logic.
- The following dunder operations on :class:`ndonnx.Array` now correctly return ``NotImplemented`` if one of the operands is not of type ``numpy.ndarray | ndonnx.Array | bool | str | int | float``: ``__add__``, ``__radd__``, ``__and__``, ``__rand__``, ``__floordiv__``, ``__rfloordiv__``, ``__ge__``, ``__gt__``, ``__le__``, ``__lshift__``, ``__rlshift__``, ``__lt__``, ``__matmul__``, ``__rmatmul__``, ``__mod__``, ``__rmod__``, ``__mul__``, ``__rmul__``, ``__or__``, ``__ror__``, ``__pow__``, ``__rpow__``, ``__rshift__``, ``__rrshift__``, ``__sub__``, ``__rsub__``, ``__truediv__``, ``__rtruediv__``, ``__xor__``, ``__rxor__``, ``__eq__``, ``__ne__``

**New workarounds for missing onnxruntime implementations**

- :func:`ndx.any` and :func:`ndx.all` now correctly propagate values even if the input is zero-sized.
- :func:`ndonnx.arange` now follows NumPy's semantics for extremely large start, stop, and step values.
- :func:`ndonnx.min`, :func:`ndonnx.max`, :func:`ndonnx.minimum`, :func:`ndonnx.maximum`, and :func:`ndonnx.clip` now produce correct results for very large values in int64 arrays.


0.13.0 (2025-05-27)
-------------------

**Bug fixes**

- :func:`ndonnx.concat` no longer raises an error if ``axis=None``, the resulting data type is ``int32`` or ``int64``, and one of the provided arrays is zero-sized.
- :func:`ndonnx.__array_namespace_info__.capabilities()` now reports the number of supported dimensions via the ``"max dimensions"`` entry rather than ``"max rank"``.
- Add missing onnxruntime workaround for ``uint32`` inputs to :func:`ndonnx.min` and :func:`ndonnx.max`.
- Fix array instantiation with :func:`ndonnx.asarray` and very large Python integers for ``uint64`` data types.
- Fix passing an Python scalar as the second argument to :func:`ndonnx.where`.
- Calling :func:`ndonnx.roll` on zero-sized inputs no longer causes a segfault on Linux.


**New features**

- The :class:`ndonnx.TimeDelta64DType` and :class:`ndonnx.DateTime64DType` gained support for milli and microseconds as units.
- :func:`ndonnx.where` now promotes time units between the two branches.
- Addition, multiplication, division, and subtraction between arrays with timedelta or datetime data types now support promotion between time units.
- Comparison operations between arrays with timedelta or datetime data types now support promotion between time units.
- Added :attr:`ndonnx.__array_api_version__` reporting the latest supported version of the Array API specification.


0.12.0 (2025-05-15)
-------------------

**Bug fix**

- The modulo operator (``%``) now correctly follows Python's semantics if the second argument is negative.


**New features**

- Support the 2024.12 version of the Array API (except ``nextafter``).


0.11.0 (2025-05-08)
-------------------

**Bug fix**

- :func:`ndonnx.mean`, :func:`ndonnx.var`, and :func:`ndonnx.std` now produce correct results if axes outside the range of `[-1, 1]` are given.


**Other change**

- Address various deprecation warnings.


0.10.1 (2025-04-01)
-------------------

Technical release to address a PyPI publishing issue.



0.10.0 (2025-04-01)
-------------------

**Breaking change**

- Removed the deprecated :func:`ndonnx.promote_nullable` function. Use :func:`ndonnx.extensions.make_nullable` instead.
- The existing (private) tooling to define custom data types has been removed.
- The :func:`ndonnx.Array.len` member function has been removed since it is neither defined by ``numpy.ndarray`` nor the Array-API.
- :func:`ndonnx.Array.size` now returns `int | None` in accordance to with the Array-API rather than a :class:`~ndonnx.Array` instance.


**Bug fixes**

- The following functions now correctly take the ``keepdims`` argument into account:
  - :meth:`ndonnx.Array.sum`
  - :meth:`ndonnx.Array.prod`
  - :meth:`ndonnx.Array.max`
  - :meth:`ndonnx.Array.min`
  - :meth:`ndonnx.Array.all`
  - :meth:`ndonnx.Array.any`


**New features**

- ndonnx gained partial support for `float16` data types.
- The following functions are now exposed in the ndonnx namespace in accordance to the Array-API standard:
  - :func:`ndonnx.__array_namespace_info__`
  - :func:`ndonnx.maximum`
  - :func:`ndonnx.minimum`
  - :func:`ndonnx.meshgrid`
  - :func:`ndonnx.moveaxis`
  - :func:`ndonnx.tile`
  - :func:`ndonnx.unstack`
  - :func:`ndonnx.vecdot`
- The newly added :mod:`ndonnx.extensions` module exposes the following functions:
  - :func:`~ndonnx.extensions.datetime_to_year_month_day`
  - :func:`~ndonnx.extensions.fill_null`
  - :func:`~ndonnx.extensions.get_mask`
  - :func:`~ndonnx.extensions.is_float_dtype`
  - :func:`~ndonnx.extensions.is_integer_dtype`
  - :func:`~ndonnx.extensions.is_nullable_dtype`
  - :func:`~ndonnx.extensions.is_nullable_float_dtype`
  - :func:`~ndonnx.extensions.is_nullable_integer_dtype`
  - :func:`~ndonnx.extensions.is_numeric_dtype`
  - :func:`~ndonnx.extensions.is_onnx_dtype`
  - :func:`~ndonnx.extensions.is_signed_integer_dtype`
  - :func:`~ndonnx.extensions.s_time_unit`
  - :func:`~ndonnx.extensions.is_unsigned_integer_dtype`
  - :func:`~ndonnx.extensions.isin`
  - :func:`~ndonnx.extensions.make_nullable`
  - :func:`~ndonnx.extensions.put`
  - :func:`~ndonnx.extensions.shape` (deprecated in favor of :func:`ndonnx.Array.dynamic_shape`)
  - :func:`~ndonnx.extensions.static_map`
- Arrays now expose the :meth:`ndonnx.Array.device` property to improve Array API compatibility. Note that serializing an ONNX model inherently postpones device placement decisions to the runtime so currently one abstract device is supported.
- The :class:`~ndonnx.Array` object gained the following member functions:
  - :func:`~ndonnx.Array.disassemble`
  - :func:`~ndonnx.Array.dynamic_shape`
  - :func:`~ndonnx.Array.dynamic_size`
  - :func:`~ndonnx.Array.unwrap_numpy`
  - :func:`~ndonnx.Array.unwrap_spox`


**Deprecations**

- :func:`ndonnx.array` is deprecated in favor of :func:`ndonnx.argument`.
- :mod:`ndonnx.additional` is deprecated in favor of :func:`ndonnx.extensions`.
- :func:`ndonnx.from_spox_var` is deprecated in favor of :func:`ndonnx.asarray`.
- :type:`ndonnx.Nullable` is deprecated in favor of :func:`ndonnx.extensions.is_nullable_dtype`
- :type:`ndonnx.NullableFloating` is deprecated in favor of :func:`ndonnx.extensions.is_nullable_float_dtype`
- :type:`ndonnx.NullableIntegral` is deprecated in favor of :func:`ndonnx.extensions.is_nullable_integer_dtype`
- :type:`ndonnx.Floating` is deprecated in favor of :func:`ndonnx.extensions.is_float_dtype`
- :type:`ndonnx.Integral` is deprecated in favor of :func:`ndonnx.extensions.is_integer_dtype`
- :type:`ndonnx.Numerical` is deprecated in favor of :func:`ndonnx.extensions.is_numeric_dtype`
- :type:`ndonnx.CoreType` is deprecated in favor of :func:`ndonnx.extensions.is_onnx_dtype`
- :type:`ndonnx.NullableCore` is deprecated in favor of :func:`ndonnx.extensions.is_nullable_dtype`
- :class:`ndonnx.UnsupportedOperationError` is deprecated in favor of :class:`TypeError`
- :class:`ndonnx.CastError` is deprecated in favor of :class:`TypeError`


**Other changes**

- :func:`~ndonnx.additional.make_nullable` can no longer be used for custom data types.


0.9.3 (2024-10-25)
------------------

- Reduced the number of unnecessary casts in :func:`ndonnx.argmax` and :func:`ndonnx.argmin`.


0.9.2 (2024-10-03)
------------------

- Technical release with source distribution.


0.9.1 (2024-10-01)
------------------

**Bug fix**

- Fixed a bug in the construction of nullable arrays using :func:`ndonnx.asarray` where the shape of the null field would not match the values field if the provided `np.ma.MaskedArray`'s mask was scalar.
- Fixed a bug in the implementation of :func:`ndonnx.ones_like` where the static shape was being used to construct the array of ones.


0.9.0 (2024-08-30)
------------------

**New features**

- User defined data types can now define how arrays with that dtype are constructed by implementing the ``make_array`` function.
- User defined data types can now define how they are indexed (via ``__getitem__``) by implementing the ``getitem`` function.
- :class:`ndonnx.NullableCore` is now public, encapsulating nullable variants of `CoreType`s exported by ndonnx.

**Bug fixes**

- Various operations that depend on the array's shape have been updated to work correctly with lazy arrays.
- :func:`ndonnx.cumulative_sum` now correctly applies the ``include_initial`` parameter and works around missing onnxruntime kernels for unsigned integral types.
- :func:`ndonnx.additional.make_nullable` applies broadcasting to the provided null array (instead of reshape like it did previously). This allows writing ``make_nullable(x, False)`` to turn an array into nullable.
- User-defined data types that implement :class:`ndonnx._core.UniformShapeOperations` may now implement :func:`ndonnx.where` without requiring both data types be promotable.

**Breaking change**

- Iterating over dynamic dimensions of :class:`~ndonnx.Array` is no longer allowed since it commonly lead to infinite loops when used without an explicit break condition.


0.8.0 (2024-08-22)
------------------

**Bug fixes**

- Fixes parsing numpy arrays of type ``object`` (consisting of strings) as ``utf8``. Previously this worked correctly only for 1d arrays.

**Breaking change**

- :meth:`ndonnx.Array.shape` now strictly returns a ``tuple[int | None, ...]``, with unknown dimensions denoted by ``None``. This relies on ONNX shape inference for lazy arrays.


0.7.0 (2024-08-12)
------------------

**New features**

- Expose the :func:`ndonnx.isdtype` function.
- Custom data types can now override array functions:
   - :func:`ndonnx.zeros`
   - :func:`ndonnx.zeros_like`
   - :func:`ndonnx.ones`
   - :func:`ndonnx.ones_like`
   - :func:`ndonnx.full`
   - :func:`ndonnx.full_like`
   - :func:`ndonnx.arange`
   - :func:`ndonnx.arange`
   - :func:`ndonnx.eye`
   - :func:`ndonnx.tril`
   - :func:`ndonnx.triu`
   - :func:`ndonnx.linspace`
   - :func:`ndonnx.where`
- The :class:`ndonnx._experimental.UniformShapeOperations` now provides implementations of shape operations that are generic across all data types where each constituent field has the same shape (that of the overall array).

**Other changes**

- Fixed various deprecation warnings.
- Invoking a function using arrays with data types that lack a corresponding implementation now raise a :class:`UnsupportedOperationError`.

**Bug fixes**

- Numerical operations like :func:`sin` now raise :class:`UnsupportedOperationError` when invoked using invalid data types like ``ndx.utf8`` rather than implicitly casting.
- Fixes bug causing a promotion error when implementing numerical operations like :func:`add` that involve type promotion.
- Fixes scalar promotion logic to more accurately reflect the Array API standard. Promotion requires at least one array to be present and scalars adopt the dtype of the arrays being promoted with it. `ndx.utf8` and `ndx.nutf8` cannot be promoted with any other dtypes.
- Fixes failure when broadcasting nullable data type arrays together in :func:`broadcast_arrays`.


0.6.1 (2024-07-12)
------------------

**Bug fixes**

- Division now complies more strictly with the Array API standard by returning a floating-point result regardless of input data types.


0.6.0 (2024-07-11)
------------------

**Other changes**

- ``ndonnx.promote_nullable`` is now publicly exported.


0.5.0 (2024-07-01)
------------------

**Other changes**

- ndonnx now exports type annotations.

**Bug fixes**

- ``__array_namespace__`` now accepts the optional ``api_version`` argument to specify the version of the Array API to use.


0.4.0 (2024-05-16)
------------------

**Breaking changes**

- The constant propagated value is no longer accessed from the ``eager_value`` property but instead the ``to_numpy()`` method.
- Non Array API functions have been moved to the ``ndonnx.additional`` namespace.
