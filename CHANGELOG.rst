.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug
Changelog
=========

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
