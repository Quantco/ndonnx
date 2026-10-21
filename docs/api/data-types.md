# Data Types

ndonnx currently exposes the following data types.
"Primitive" data types have a directly corresponding data type in the ONNX standard.

| Name          | Description                                                      | Array API | ONNX primitive |
| ------------- | ---------------------------------------------------------------- | --------- | -------------- |
| `float16`     | 16-bit floating point number                                     | Yes       | Yes            |
| `float32`     | 32-bit floating point number                                     | Yes       | Yes            |
| `float64`     | 64-bit floating point number                                     | Yes       | Yes            |
| `int8`        | 8-bit signed integer                                             | Yes       | Yes            |
| `int16`       | 16-bit signed integer                                            | Yes       | Yes            |
| `int32`       | 32-bit signed integer                                            | Yes       | Yes            |
| `int64`       | 64-bit signed integer                                            | Yes       | Yes            |
| `uint8`       | 8-bit unsigned integer                                           | Yes       | Yes            |
| `uint16`      | 16-bit unsigned integer                                          | Yes       | Yes            |
| `uint32`      | 32-bit unsigned integer                                          | Yes       | Yes            |
| `uint64`      | 64-bit unsigned integer                                          | Yes       | Yes            |
| `bool`        | Boolean                                                          | Yes       | Yes            |
| `utf8`        | UTF-8 string                                                     | No        | Yes            |
| `nfloat16`    | Nullable 16-bit floating point number                            | No        | No             |
| `nfloat32`    | Nullable 32-bit floating point number                            | No        | No             |
| `nfloat64`    | Nullable 64-bit floating point number                            | No        | No             |
| `nint8`       | Nullable 8-bit signed integer                                    | No        | No             |
| `nint16`      | Nullable 16-bit signed integer                                   | No        | No             |
| `nint32`      | Nullable 32-bit signed integer                                   | No        | No             |
| `nint64`      | Nullable 64-bit signed integer                                   | No        | No             |
| `nuint8`      | Nullable 8-bit unsigned integer                                  | No        | No             |
| `nuint16`     | Nullable 16-bit unsigned integer                                 | No        | No             |
| `nuint32`     | Nullable 32-bit unsigned integer                                 | No        | No             |
| `nuint64`     | Nullable 64-bit unsigned integer                                 | No        | No             |
| `nbool`       | Nullable boolean                                                 | No        | No             |
| `nutf8`       | Nullable UTF-8 string                                            | No        | No             |
| `datetime64`  | Datetime data type that follows NumPy's `datetime64` semantics   | No        | No             |
| `timedelta64` | Timedelta data type that follows NumPy's `timedelta64` semantics | No        | No             |

## Data type objects

<!-- prettier-ignore -->
::: ndonnx
    options:
      show_root_members_full_path: true
      members:
        - bool
        - float16
        - float32
        - float64
        - int8
        - int16
        - int32
        - int64
        - uint8
        - uint16
        - uint32
        - uint64
        - utf8
        - nbool
        - nfloat16
        - nfloat32
        - nfloat64
        - nint8
        - nint16
        - nint32
        - nint64
        - nuint8
        - nuint16
        - nuint32
        - nuint64
        - nutf8
        - DateTime64DType
        - TimeDelta64DType

## DType

<!-- prettier-ignore -->
::: ndonnx.DType
    options:
      summary: true
      filters:
        - "!^_"
