Data Types
==========

ndonnx exposes the following data types.
There is also :doc:`unstable support <../experimental/experimental>` for defining your own structured data types.
"Primitive" data types have a directly corresponding data type in the ONNX standard.


.. list-table::
    :widths: 25 50 20 20
    :header-rows: 1

    * - Name
      - Description
      - Array API
      - ONNX primitive
    * - ``float32``
      - 32-bit floating point number
      - Yes
      - Yes
    * - ``float64``
      - 64-bit floating point number
      - Yes
      - Yes
    * - ``int8``
      - 8-bit signed integer
      - Yes
      - Yes
    * - ``int16``
      - 16-bit signed integer
      - Yes
      - Yes
    * - ``int32``
      - 32-bit signed integer
      - Yes
      - Yes
    * - ``int64``
      - 64-bit signed integer
      - Yes
      - Yes
    * - ``uint8``
      - 8-bit unsigned integer
      - Yes
      - Yes
    * - ``uint16``
      - 16-bit unsigned integer
      - Yes
      - Yes
    * - ``uint32``
      - 32-bit unsigned integer
      - Yes
      - Yes
    * - ``uint64``
      - 64-bit unsigned integer
      - Yes
      - Yes
    * - ``bool``
      - Boolean
      - Yes
      - Yes
    * - ``utf8``
      - UTF-8 string
      - No
      - Yes
    * - ``nfloat32``
      - Nullable 32-bit floating point number
      - No
      - No
    * - ``nfloat64``
      - Nullable 64-bit floating point number
      - No
      - No
    * - ``nint8``
      - Nullable 8-bit signed integer
      - No
      - No
    * - ``nint16``
      - Nullable 16-bit signed integer
      - No
      - No
    * - ``nint32``
      - Nullable 32-bit signed integer
      - No
      - No
    * - ``nint64``
      - Nullable 64-bit signed integer
      - No
      - No
    * - ``nuint8``
      - Nullable 8-bit unsigned integer
      - No
      - No
    * - ``nuint16``
      - Nullable 16-bit unsigned integer
      - No
      - No
    * - ``nuint32``
      - Nullable 32-bit unsigned integer
      - No
      - No
    * - ``nuint64``
      - Nullable 64-bit unsigned integer
      - No
      - No
    * - ``nbool``
      - Nullable boolean
      - No
      - No
    * - ``nutf8``
      - Nullable UTF-8 string
      - No
      - No
    * - ``datetime64``
      - Datetime data type that follows NumPy's ``datetime64`` semantics
      - No
      - No
    * - ``timedelta64``
      - Timedelta data type that follows NumPy's ``timedelta64`` semantics
      - No
      - No
