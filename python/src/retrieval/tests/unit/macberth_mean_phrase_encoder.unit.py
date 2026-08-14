encoder = MacBertMeanPhraseEncoder(
    tokenizer,
    model,
)

vector = encoder.encode(
    "extreme whiteness"
)

assert vector.dtype == np.float32
assert vector.shape == (768,)
assert np.isclose(
    np.linalg.norm(vector),
    1.0,
    atol=1e-5,
)
