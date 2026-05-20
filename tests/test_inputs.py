from hebbwatch.inputs.generators import build_input_generator


def test_input_generator_shape():
    gen = build_input_generator("correlated", 6, seed=1)
    x = next(gen)
    assert x.shape == (6,)
