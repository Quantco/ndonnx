.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

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
