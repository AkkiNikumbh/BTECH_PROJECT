"""2021-2023 cycle mapping (pre-registered, docs/analysis_plan_v5_confirmatory.md)."""
import numpy as np


def test_pad810q_maps_to_paq650_coding():
    from src.data_loader import pad810q_to_paq650
    out = pad810q_to_paq650([0, 1, 3, 200, 7777, 9999, np.nan])
    assert list(out[:4]) == [2.0, 1.0, 1.0, 1.0]
    assert np.isnan(out[4:]).all()
