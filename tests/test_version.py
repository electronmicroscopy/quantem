import quantem


def test_version_is_string():
    assert isinstance(quantem.__version__, str)
