import hmmsort
import hmmsort.utility as util
import pytest


def test_first():
    assert util.first([1]) == 1
    assert util.first(1) == 1
    with pytest.raises(Exception):
        assert util.first([])
