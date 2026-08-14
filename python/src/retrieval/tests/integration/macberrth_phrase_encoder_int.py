query = encoder.encode(
    "extreme whiteness"
)

result = diskann.search(
    query.reshape(1, -1),
    k=20,
)