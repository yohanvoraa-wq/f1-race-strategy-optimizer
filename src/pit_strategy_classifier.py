# ============================================================
# MODEL 2 — RACE STRATEGY OPTIMIZATION
# F1 ML Project | NMIMS STME | Semester 4
# Team: Yohan Vora (A186) | Jugraj Singh (A188)
# Submitted To: Dr. Divyang Jadav
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_classifier(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n--- {name} ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred,
          target_names=['Stay Out', 'Pit']))
    return acc, prec, rec, f1

# ============================================================
# STEP 1 — LOAD ALL 5 CIRCUITS
# ============================================================

races = ['Bahrain', 'Azerbaijan', 'British', 'Italian', 'Singapore']

dfs = []
for race in races:
    df = pd.read_csv(f'f1_data/race_wise/{race}/strategy_{race}.csv.csv')
    df['RaceName'] = race
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Total rows loaded: {len(data)}")
print(f"Columns: {data.columns.tolist()}")

# ============================================================
# STEP 2 — DATA CLEANING
# ============================================================

print(f"\nRows before cleaning: {len(data)}")
data = data.dropna(subset=['LapTime_s'])
print(f"Rows after dropping missing LapTime: {len(data)}")
data = data.dropna(subset=['DegRate_s_lap'])
print(f"Rows after dropping missing DegRate: {len(data)}")
valid_compounds = ['SOFT', 'MEDIUM', 'HARD']
data = data[data['Compound'].isin(valid_compounds)]
print(f"Rows after compound filter: {len(data)}")

# ============================================================
# STEP 3 — CLASS DISTRIBUTION
# ============================================================

pit_counts = data['IsPitLap'].value_counts()
total      = len(data)
print(f"\n=== CLASS DISTRIBUTION ===")
print(f"Stay Out (0) : {pit_counts[0]} ({round(pit_counts[0]/total*100,2)}%)")
print(f"Pit      (1) : {pit_counts[1]} ({round(pit_counts[1]/total*100,2)}%)")
print(f"Imbalance ratio: {round(pit_counts[0]/pit_counts[1],1)}:1")

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='IsPitLap', palette=['green', 'red'])
plt.xticks([0, 1], ['Stay Out (0)', 'Pit (1)'])
plt.title('Class Distribution — Pit vs Stay Out')
plt.xlabel('Decision')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('model2_class_distribution.png', dpi=150)
plt.show()

# ============================================================
# STEP 4 — FEATURE ENGINEERING AND ENCODING
# ============================================================

# Encode Compound
le = LabelEncoder()
data['Compound_enc'] = le.fit_transform(data['Compound'])
print(f"\nCompound encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# One-hot encode Race
data = pd.get_dummies(data, columns=['Race'], prefix='Race')
race_columns = [col for col in data.columns if col.startswith('Race_')]
print(f"Race columns: {race_columns}")

# Convert race dummy columns to integer
for col in race_columns:
    data[col] = data[col].astype(int)

# Verify race columns
print("\nRace column verification:")
for circuit in races:
    race_col = f'Race_{circuit}'
    if race_col in data.columns:
        count   = data[data[race_col] == 1].shape[0]
        avg_deg = data[data[race_col] == 1]['DegRate_s_lap'].mean()
        print(f"  {circuit}: {count} rows, avg DegRate = {avg_deg:.4f} s/lap")

# ============================================================
# FIX — CORRECT NEGATIVE DEGRADATION RATES
# ============================================================

print(f"\nDegRate before fix: min={data['DegRate_s_lap'].min():.4f}, "
      f"max={data['DegRate_s_lap'].max():.4f}, "
      f"mean={data['DegRate_s_lap'].mean():.4f}")

# Flip sign if majority of values are negative
if data['DegRate_s_lap'].mean() < 0:
    data['DegRate_s_lap'] = data['DegRate_s_lap'].abs()
    print("DegRate sign corrected — values flipped to positive")

# Remove unrealistic degradation values (real F1 deg: 0.01 to 1.0 s/lap)
data = data[(data['DegRate_s_lap'] >= 0.01) &
            (data['DegRate_s_lap'] <= 1.0)]
print(f"Rows after DegRate sanity filter: {len(data)}")

print(f"DegRate after fix: min={data['DegRate_s_lap'].min():.4f}, "
      f"max={data['DegRate_s_lap'].max():.4f}, "
      f"mean={data['DegRate_s_lap'].mean():.4f}")

print("\nDegRate per circuit after fix:")
for circuit in races:
    race_col = f'Race_{circuit}'
    if race_col in data.columns:
        subset  = data[data[race_col] == 1]
        avg_deg = subset['DegRate_s_lap'].mean()
        print(f"  {circuit}: avg DegRate = {avg_deg:.4f} s/lap")

# Final feature set
features = [
    'LapNumber',
    'Compound_enc',
    'TyreLife',
    'FuelLoad_kg',
    'FuelCorrection_s',
    'DegRate_s_lap',
    'LapsRemaining',
    'Position',
    'SafetyCar',
    'RedFlag',
] + race_columns

target = 'IsPitLap'

X = data[features]
y = data[target]

print(f"\nFeature matrix shape: {X.shape}")

# ============================================================
# STEP 5 — TRAIN/TEST SPLIT (STRATIFIED)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")
print(f"Training pit laps : {y_train.sum()} ({round(y_train.sum()/len(y_train)*100,2)}%)")
print(f"Testing pit laps  : {y_test.sum()} ({round(y_test.sum()/len(y_test)*100,2)}%)")

# ============================================================
# STEP 6 — MODEL TRAINING
# ============================================================

# --- Baseline: Decision Tree ---
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_acc, dt_prec, dt_rec, dt_f1 = evaluate_classifier(
    "Decision Tree", y_test, dt_preds)

# --- Primary: XGBoost with GridSearch ---
scale = int(y_train.value_counts()[0] / y_train.value_counts()[1])
print(f"\nXGBoost scale_pos_weight: {scale}")

param_grid = {
    'max_depth'        : [3, 4, 5],
    'learning_rate'    : [0.01, 0.05, 0.1],
    'n_estimators'     : [100, 200, 300],
    'scale_pos_weight' : [20, 32, 40, 50]
}

xgb_base = XGBClassifier(
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    eval_metric='logloss'
)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"\nBest Parameters : {grid_search.best_params_}")
print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

xgb_model = grid_search.best_estimator_
xgb_preds = xgb_model.predict(X_test)
xgb_acc, xgb_prec, xgb_rec, xgb_f1 = evaluate_classifier(
    "XGBoost Tuned", y_test, xgb_preds)

# ============================================================
# STEP 7 — CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Model 2 — Race Strategy Confusion Matrices',
             fontsize=14, fontweight='bold')

for ax, preds, name, color in zip(
    axes,
    [dt_preds, xgb_preds],
    ['Decision Tree', 'XGBoost Tuned'],
    ['Blues', 'Reds']
):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                xticklabels=['Stay Out', 'Pit'],
                yticklabels=['Stay Out', 'Pit'])
    ax.set_title(f'{name}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('model2_confusion_matrices.png', dpi=150)
plt.show()
print("Confusion matrices saved")

# ============================================================
# STEP 8 — DYNAMIC PROGRAMMING BREAKEVEN ANALYSIS
# ============================================================

print("\n" + "="*72)
print("DYNAMIC PROGRAMMING — PIT STOP BREAKEVEN ANALYSIS")
print("="*72)

pit_loss_times = {
    'Azerbaijan' : 24.0,
    'Bahrain'    : 22.0,
    'British'    : 20.0,
    'Italian'    : 21.0,
    'Singapore'  : 28.0
}

def breakeven_analysis(deg_rate, laps_remaining, pit_loss):
    if deg_rate <= 0:
        return None
    breakeven_laps = pit_loss / deg_rate
    time_gained    = deg_rate * laps_remaining
    recommendation = "PIT" if time_gained > pit_loss else "STAY OUT"
    return breakeven_laps, time_gained, recommendation

print(f"\n{'Circuit':<12} {'Pit Loss':>10} {'Avg DegRate':>12} "
      f"{'Breakeven':>12} {'Time Gained':>12} {'Decision':>12}")
print("-"*72)

for circuit, pit_loss in pit_loss_times.items():
    race_col = f'Race_{circuit}'
    subset   = data[data[race_col] == 1]
    avg_deg  = subset['DegRate_s_lap'].mean()
    avg_laps = subset['LapsRemaining'].mean()
    result   = breakeven_analysis(avg_deg, avg_laps, pit_loss)
    if result:
        be_laps, time_gained, rec = result
        print(f"{circuit:<12} {pit_loss:>9.1f}s "
              f"{avg_deg:>11.4f}s "
              f"{be_laps:>11.1f}L "
              f"{time_gained:>11.2f}s "
              f"{rec:>12}")

print("="*72)
print("\nINTERPRETATION:")
print("Breakeven  = minimum laps needed on fresh tires to justify a pit stop")
print("Time Gained = total seconds saved by pitting (DegRate x LapsRemaining)")
print("Decision   = PIT if Time Gained > Pit Loss Time, else STAY OUT")

# ============================================================
# STEP 9 — RESULTS SUMMARY
# ============================================================

print("\n" + "="*55)
print("MODEL 2 — FINAL RESULTS SUMMARY")
print("="*55)
results_df = pd.DataFrame({
    'Model'     : ['Decision Tree', 'XGBoost Tuned'],
    'Accuracy'  : [round(dt_acc,4),  round(xgb_acc,4)],
    'Precision' : [round(dt_prec,4), round(xgb_prec,4)],
    'Recall'    : [round(dt_rec,4),  round(xgb_rec,4)],
    'F1 Score'  : [round(dt_f1,4),   round(xgb_f1,4)]
})
print(results_df.to_string(index=False))
print("="*55)

# ============================================================
# STEP 10 — SAVE MODEL
# ============================================================

joblib.dump(xgb_model, 'saved_models/model2_xgb_strategy.pkl')
joblib.dump(le,        'saved_models/model2_label_encoder.pkl')
joblib.dump(features,  'saved_models/model2_features.pkl')
print("\nModel 2 saved successfully ✅")

# ============================================================
# STEP 11 — STRATEGY PREDICTOR (USER INPUT)
# ============================================================

print("\n" + "="*60)
print("RACE STRATEGY PREDICTOR — ENTER YOUR SCENARIO")
print("="*60)
print("\nEXAMPLE INPUT (use these values to test):")
print("  LapNumber     : 25")
print("  Compound      : MEDIUM")
print("  TyreLife      : 18")
print("  FuelLoad_kg   : 45.0")
print("  DegRate_s_lap : 0.09")
print("  LapsRemaining : 35")
print("  Position      : 5")
print("  SafetyCar     : 0")
print("  RedFlag       : 0")
print("  Race          : Bahrain")
print("="*60)

print("\nEnter your scenario:\n")

lap_number     = float(input("Enter LapNumber (e.g. 25): "))
compound_input = input("Enter Compound (SOFT / MEDIUM / HARD): ").strip().upper()
while compound_input not in ['SOFT', 'MEDIUM', 'HARD']:
    compound_input = input("Invalid. Enter SOFT / MEDIUM / HARD: ").strip().upper()
compound_enc   = le.transform([compound_input])[0]
tyre_life      = float(input("Enter TyreLife — laps on current tires (e.g. 18): "))
fuel_load      = float(input("Enter FuelLoad_kg (e.g. 45.0): "))
fuel_correction= round(fuel_load * 0.03, 3)
deg_rate       = float(input("Enter DegRate_s_lap — tire degradation per lap (e.g. 0.09): "))
laps_remaining = float(input("Enter LapsRemaining (e.g. 35): "))
position       = float(input("Enter Position (1–20): "))
safety_car     = int(input("SafetyCar active? (0 = No, 1 = Yes): "))
red_flag       = int(input("RedFlag active? (0 = No, 1 = Yes): "))
race_input     = input("Enter Race (Bahrain / Azerbaijan / British / Italian / Singapore): ").strip().capitalize()
while race_input not in races:
    race_input = input(f"Invalid. Choose from {races}: ").strip().capitalize()

# Build scenario
scenario = {
    'LapNumber'        : lap_number,
    'Compound_enc'     : compound_enc,
    'TyreLife'         : tyre_life,
    'FuelLoad_kg'      : fuel_load,
    'FuelCorrection_s' : fuel_correction,
    'DegRate_s_lap'    : deg_rate,
    'LapsRemaining'    : laps_remaining,
    'Position'         : position,
    'SafetyCar'        : safety_car,
    'RedFlag'          : red_flag,
    'Race_Azerbaijan'  : 1 if race_input == 'Azerbaijan' else 0,
    'Race_Bahrain'     : 1 if race_input == 'Bahrain'    else 0,
    'Race_British'     : 1 if race_input == 'British'    else 0,
    'Race_Italian'     : 1 if race_input == 'Italian'    else 0,
    'Race_Singapore'   : 1 if race_input == 'Singapore'  else 0,
}

scenario_df = pd.DataFrame([scenario])

# Predict
dt_decision  = dt_model.predict(scenario_df)[0]
xgb_decision = xgb_model.predict(scenario_df)[0]
dt_label     = "PIT" if dt_decision  == 1 else "STAY OUT"
xgb_label    = "PIT" if xgb_decision == 1 else "STAY OUT"

# Breakeven check
pit_loss = pit_loss_times.get(race_input, 22.0)
result   = breakeven_analysis(deg_rate, laps_remaining, pit_loss)

print("\n" + "="*60)
print("STRATEGY PREDICTION RESULTS")
print("="*60)
print(f"  Circuit       : {race_input} Grand Prix")
print(f"  Lap Number    : {int(lap_number)}")
print(f"  Compound      : {compound_input} (Age: {int(tyre_life)} laps)")
print(f"  Position      : P{int(position)}")
print(f"  Laps Remaining: {int(laps_remaining)}")
print(f"  Safety Car    : {'Yes' if safety_car else 'No'}")
print(f"")
print(f"  Decision Tree : {dt_label}")
print(f"  XGBoost       : {xgb_label}")
print(f"")
if result:
    be_laps, time_gained, dp_rec = result
    print(f"  Dynamic Programming Analysis:")
    print(f"  Pit Loss Time    : {pit_loss}s")
    print(f"  Time Gained      : {time_gained:.2f}s")
    print(f"  Breakeven Point  : {be_laps:.1f} laps")
    print(f"  DP Recommendation: {dp_rec}")

# Human readable explanation
wear_level = (
    "fresh"        if tyre_life <= 5  else
    "lightly used" if tyre_life <= 15 else
    "heavily worn"
)
sc_note = " A Safety Car is on track —" if safety_car else ""
print(f"")
print(f"  WHAT THIS MEANS:")
print(f"  Your driver is on {wear_level} {compound_input} tires")
print(f"  (used for {int(tyre_life)} laps) at Lap {int(lap_number)}{sc_note}")
print(f"  with {int(laps_remaining)} laps remaining at the {race_input} GP.")
print(f"  The model recommends: {xgb_label}")
print("="*60)
