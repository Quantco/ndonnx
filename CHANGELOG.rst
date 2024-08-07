.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug

Changelog
=========

0.7.0 (2024-08-07)
------------------

**New feature**

- Expose the :func:`ndonnx.isdtype` function.

**Other changes**

- Fixed various deprecation warnings.

**Bug fix**

- Fixes scalar promotion logic to more accurately reflect the Array API standard. Promotion requires at least one array to be present and scalars adopt the dtype of the arrays being promoted with it. ``ndx.utf8`` and ``ndx.nutf8`` cannot be promoted with any other data type.


0.6.1 (2024-07-12)
------------------

**Bug fix**

- Division now complies more strictly with the Array API standard by returning a floating-point result regardless of input data types.


0.6.0 (2024-07-11)
------------------

**Other changes**

- ``ndonnx.promote_nullable`` is now publicly exported.


0.5.0 (2024-07-01)
------------------

**Other changes**

- ndonnx now exports type annotations.

**Bug fix**

- ``__array_namespace__`` now accepts the optional ``api_version`` argument to specify the version of the Array API to use.


0.4.0 (2024-05-16)
------------------

**Breaking changes**

- The constant propagated value is no longer accessed from the ``eager_value`` property but instead the ``to_numpy()`` method.
- Non-Array API functions have been moved to the ``ndonnx.additional`` namespace.
