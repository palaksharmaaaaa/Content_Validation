import numpy as np

from core.metrics_util import sanitize_metric_value as sv


def test_scalars_arrays_and_nesting():
    assert sv(1.5) == 1.5 and sv("a") == "a"
    assert sv(np.float32(2.0)) == 2.0
    assert sv(np.zeros((2, 2))) is None
    assert sv({"a": 1, "b": np.zeros(3), 2: [1, np.ones(2), {"c": 3}]}) == {"a": 1, "2": [1, {"c": 3}]}
    assert sv(object()) is None
