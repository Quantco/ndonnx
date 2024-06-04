# Create concatenation of all signature files, removing the definitions of array
sed '$s/$/\n/' api-coverage-tests/array-api/spec/API_specification/signatures/*.py | sed '/^array = /d' > ndonnx/__init__.pyi

# Create _types.py
echo "from ._array import Array as array\n$(<api-coverage-tests/array-api/spec/API_specification/signatures/_types.py)" | sed '/^array = /d' > ndonnx/_types.pyi
