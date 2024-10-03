.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

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
