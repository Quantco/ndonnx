.. Versioning follows semantic versioning, see also
   https://semver.org/spec/v2.0.0.html. The most important bits are:
   * Update the major if you break the public API
   * Update the minor if you add new functionality
   * Update the patch if you fixed a bug
Changelog
=========

0.4.0 (2024-05-16)
------------------

**Breaking changes**

- The constant propagated value is no longer accessed from the ``eager_value`` property but instead the ``to_numpy()`` method.
- Non Array API functions have been moved to the ``ndonnx.additional`` namespace.
