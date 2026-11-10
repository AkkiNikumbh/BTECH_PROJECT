"""
Stage 3: Merge NHANES files, clean data, compute reporting gaps,
define lie budget, and prepare both feature sets.
Usage: python scripts/prepare_data.py
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import set_all_seeds, DATA_PROCESSED
from src.data_loader import merge_nhanes, clean_nhanes, create_diabetes_target, prepare_feature_sets
from src.gap_analysis import compute_reporting_gap, define_lie_budget, plot_reporting_gaps

set_all_seeds()
os.makedirs(DATA_PROCESSED, exist_ok=True)

print("="*60)
print("  Stage 3: Data Preparation")
print("  NHANES 2015-2016 (development set only)")
print("="*60)

# Step 1: Merge
df = merge_nhanes(cycle='2015_2016')

# Step 2: Clean
df = clean_nhanes(df)

# Step 3: Target
df = create_diabetes_target(df)

# Step 4: Gap analysis -> lie budget
df = compute_reporting_gap(df)
lie_budget = define_lie_budget(df, method='percentile', pct=95)

# Step 5: Plot gaps
plot_reporting_gaps(df, save_dir=os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'results', 'figures'))

# Step 6: Feature sets
X_meas, X_rep, y, fn_meas, fn_rep, df_complete = prepare_feature_sets(df)

# Step 7: Save
output = {
    'X_measured'          : X_meas,
    'X_reported'          : X_rep,
    'y'                   : y,
    'feature_names_measured': fn_meas,
    'feature_names_reported': fn_rep,
    'lie_budget'          : lie_budget,
    'df_complete'         : df_complete,
}
save_path = os.path.join(DATA_PROCESSED, 'nhanes_2015_2016_prepared.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(output, f)

print(f"\n[DONE] Saved processed data -> {save_path}")
print(f"       Patients : {len(y)}")
print(f"       Lie budget: height=+-{lie_budget['height']['max_lie']:.2f}cm  "
      f"weight=+-{lie_budget['weight']['max_lie']:.2f}kg")
