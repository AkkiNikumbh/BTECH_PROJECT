# -*- coding: utf-8 -*-
"""
Download relevant NHANES data files for both survey rounds.
Usage: python scripts/download_nhanes.py
"""
import os
import sys
import urllib.request

# Base URLs for NHANES data (updated CDC URL structure)
BASE_2015 = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/"
BASE_2017 = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"

MIN_VALID_SIZE = 50 * 1024   # Anything < 50 KB is an HTML error page, not real XPT

# Files to download: (filename, description, cycle_suffix)
FILES = [
    # Demographics
    ("DEMO_I.XPT", "Demographics (2015-2016)", "I"),
    ("DEMO_J.XPT", "Demographics (2017-2018)", "J"),

    # Body Measures (Examination - nurse measured)
    ("BMX_I.XPT", "Body Measures Exam (2015-2016)", "I"),
    ("BMX_J.XPT", "Body Measures Exam (2017-2018)", "J"),

    # Weight History (Questionnaire - self reported)
    ("WHQ_I.XPT", "Weight History Questionnaire (2015-2016)", "I"),
    ("WHQ_J.XPT", "Weight History Questionnaire (2017-2018)", "J"),

    # Glycohemoglobin (Lab - HbA1c for diabetes)
    ("GHB_I.XPT", "Glycohemoglobin / HbA1c (2015-2016)", "I"),
    ("GHB_J.XPT", "Glycohemoglobin / HbA1c (2017-2018)", "J"),

    # Blood Pressure (Examination)
    ("BPX_I.XPT", "Blood Pressure Exam (2015-2016)", "I"),
    ("BPX_J.XPT", "Blood Pressure Exam (2017-2018)", "J"),

    # Smoking (Questionnaire)
    ("SMQ_I.XPT", "Smoking Questionnaire (2015-2016)", "I"),
    ("SMQ_J.XPT", "Smoking Questionnaire (2017-2018)", "J"),

    # Physical Activity (Questionnaire)
    ("PAQ_I.XPT", "Physical Activity Questionnaire (2015-2016)", "I"),
    ("PAQ_J.XPT", "Physical Activity Questionnaire (2017-2018)", "J"),
]


def download_nhanes_files(output_base=None):
    """Download all listed NHANES XPT files."""
    if output_base is None:
        output_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

    success = 0
    skipped = 0
    failed = 0

    for filename, description, suffix in FILES:
        base_url = BASE_2015 if suffix == 'I' else BASE_2017
        cycle = '2015_2016' if suffix == 'I' else '2017_2018'
        subdir = os.path.join(output_base, f'nhanes_{cycle}')
        os.makedirs(subdir, exist_ok=True)

        filepath = os.path.join(subdir, filename)

        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size > MIN_VALID_SIZE:
                print(f"  [SKIP] {description} — already exists ({size//1024} KB)")
                skipped += 1
                continue
            else:
                print(f"  [REDOWNLOAD] {description} — existing file too small ({size} bytes), re-fetching")
                os.remove(filepath)

        url = base_url + filename
        print(f"  [DOWNLOAD] {description}")
        print(f"    URL: {url}")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"    >> Saved ({size_mb:.1f} MB)")
            success += 1
        except Exception as e:
            print(f"    >> ERROR: {e}")
            failed += 1

    print(f"\nDone: {success} downloaded, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    print("=" * 60)
    print("  NHANES Data Download Script")
    print("  GP-109 BTP Project")
    print("=" * 60)
    download_nhanes_files()
