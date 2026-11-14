"""
Research plan P1: the pipeline reproduces by-hand results on UCI Heart
Disease (UCI 45). Skipped if the dataset has not been downloaded
(data/raw/uci_45_heart_disease/heart_disease.zip).
"""
import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from src.supporting_data import UCI45_ZIP


@pytest.mark.skipif(not os.path.exists(UCI45_ZIP), reason='UCI 45 not downloaded')
def test_uci_heart_hand_check():
    from check_uci_heart import run_hand_check
    checks = run_hand_check(verbose=False)
    failed = [k for k, ok in checks.items() if not ok]
    assert not failed, failed
