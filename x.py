import ndonnx._refactor as ndx

# FIXME: I have to cast this to my custom dtype, as there is no way to define this for my custom dtype.
x = ndx.asarray(["abc", "def"], dtype=ndx.utf8).astype(ndx.object_dtype)
# x = ndx.asarray(["abc", "def"], dtype=ndx.object_dtype)
# illustrates why we need to be able to do the obove statement. This will crash since, appropriately, None is not a string.
# y = ndx.asarray([None, "ghi"], dtype=ndx.utf8).astype(ndx.object_dtype)
print(x)
