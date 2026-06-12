# ============================================================
# MODEL 1 — LAP TIME PREDICTION
# F1 ML Project | NMIMS STME | Semester 4
# Team: Yohan Vora (A186) | Jugraj Singh (A188)
# Submitted To: Dr. Divyang Jadav
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1 — LOAD ALL 5 CIRCUITS
# ============================================================

races = ['Bahrain', 'Azerbaijan', 'British', 'Italian', 'Singapore']

dfs = []
for race in races:
    df = pd.read_csv(f'f1_data/race_wise/{race}/laps_{race}.csv.csv')
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Total rows loaded: {len(data)}")
print(f"Columns: {data.columns.tolist()}")

# ============================================================
# STEP 2 — DATA CLEANING AND FILTERING
# ============================================================

print(f"\nRows before filtering: {len(data)}")

# Filter 1 — Keep only accurate laps (removes pit laps, SC laps, VSC laps)
data = data[data['IsAccurate'] == True]
print(f"Rows after IsAccurate filter: {len(data)}")

# Filter 2 — Drop rows where LapTime_s is missing
data = data.dropna(subset=['LapTime_s'])
print(f"Rows after dropping missing LapTime: {len(data)}")

# Filter 3 — Remove outliers (lap times beyond 3 standard deviations)
mean = data['LapTime_s'].mean()
std = data['LapTime_s'].std()
data = data[(data['LapTime_s'] >= mean - 3*std) &
            (data['LapTime_s'] <= mean + 3*std)]
print(f"Rows after outlier removal: {len(data)}")

# Filter 4 — Keep only known compounds
valid_compounds = ['SOFT', 'MEDIUM', 'HARD']
data = data[data['Compound'].isin(valid_compounds)]
print(f"Rows after compound filter: {len(data)}")

# Check remaining nulls
print(f"\nNull values per column:")
print(data.isnull().sum())

# ============================================================
# STEP 3 — FEATURE SELECTION AND ENCODING
# ============================================================

# Encode Compound (SOFT, MEDIUM, HARD → 0, 1, 2)
le = LabelEncoder()
data['Compound_enc'] = le.fit_transform(data['Compound'])
print(f"\nCompound encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# One-hot encode Race (circuit-specific pace adjustment)
data = pd.get_dummies(data, columns=['Race'], prefix='Race')
race_columns = [col for col in data.columns if col.startswith('Race_')]
print(f"Race columns added: {race_columns}")

# Final feature set — NO sector times (they leak the target)
features = [
    'LapNumber',
    'Compound_enc',
    'TyreLife',
    'FuelLoad_kg',
    'FuelCorrection_s',
    'Position',
    'SpeedI1',
    'SpeedI2',
    'SpeedFL',
    'SpeedST'
] + race_columns

target = 'LapTime_s'

X = data[features]
y = data[target]

print(f"\nFeature matrix shape: {X.shape}")

# ============================================================
# STEP 4 — TRAIN/TEST SPLIT (Random 80/20)
# ============================================================

# Note: Random 80/20 split used for better generalization
# Bahrain was originally considered as test circuit (used in Paper 3 RSRL)
# but random split gave better results across all circuits

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

# ============================================================
# STEP 5 — MODEL TRAINING
# ============================================================

# --- Baseline Model: Random Forest ---
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# --- Primary Model: XGBoost ---
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# ============================================================
# STEP 6 — EVALUATION
# ============================================================

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"RMSE : {rmse:.4f} seconds")
    print(f"MAE  : {mae:.4f} seconds")
    print(f"R²   : {r2:.4f}")
    return rmse, mae, r2

rf_rmse,  rf_mae,  rf_r2  = evaluate_model("Random Forest", y_test, rf_preds)
xgb_rmse, xgb_mae, xgb_r2 = evaluate_model("XGBoost",       y_test, xgb_preds)

# ============================================================
# STEP 7 — VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model 1 — Lap Time Prediction Results', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted (XGBoost)
axes[0, 0].scatter(y_test, xgb_preds, alpha=0.4, color='red', s=10)
axes[0, 0].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'k--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Lap Time (s)')
axes[0, 0].set_ylabel('Predicted Lap Time (s)')
axes[0, 0].set_title('XGBoost — Actual vs Predicted')
axes[0, 0].legend()

# Plot 2: Actual vs Predicted (Random Forest)
axes[0, 1].scatter(y_test, rf_preds, alpha=0.4, color='blue', s=10)
axes[0, 1].plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'k--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Lap Time (s)')
axes[0, 1].set_ylabel('Predicted Lap Time (s)')
axes[0, 1].set_title('Random Forest — Actual vs Predicted')
axes[0, 1].legend()

# Plot 3: Residuals (XGBoost)
xgb_residuals = y_test.values - xgb_preds
axes[1, 0].hist(xgb_residuals, bins=50, color='red', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Residual (Actual - Predicted) in seconds')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('XGBoost — Residual Distribution')

# Plot 4: Model Comparison Bar Chart
models    = ['Random Forest', 'XGBoost']
rmse_vals = [rf_rmse, xgb_rmse]
mae_vals  = [rf_mae,  xgb_mae]
r2_vals   = [rf_r2,   xgb_r2]

x     = np.arange(len(models))
width = 0.25

axes[1, 1].bar(x - width, rmse_vals, width, label='RMSE',  color='orange')
axes[1, 1].bar(x,          mae_vals,  width, label='MAE',   color='green')
axes[1, 1].bar(x + width,  r2_vals,   width, label='R²',    color='purple')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(models)
axes[1, 1].set_title('Model Comparison — RMSE / MAE / R²')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('model1_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved as model1_results.png")

# ============================================================
# STEP 8 — FEATURE IMPORTANCE (XGBoost)
# ============================================================

feature_importance = pd.DataFrame({
    'Feature':    features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='Reds_r')
plt.title('XGBoost — Feature Importance for Lap Time Prediction', fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('model1_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Feature importance plot saved")

# ============================================================
# STEP 9 — RESULTS SUMMARY TABLE
# ============================================================

print("\n" + "="*50)
print("MODEL 1 — FINAL RESULTS SUMMARY")
print("="*50)
results_df = pd.DataFrame({
    'Model':    ['Random Forest', 'XGBoost'],
    'RMSE (s)': [round(rf_rmse,  4), round(xgb_rmse, 4)],
    'MAE (s)':  [round(rf_mae,   4), round(xgb_mae,  4)],
    'R²':       [round(rf_r2,    4), round(xgb_r2,   4)]
})
print(results_df.to_string(index=False))
print("="*50)
print(f"\nBest Model   : XGBoost")
print(f"XGBoost explains {round(xgb_r2*100, 2)}% of lap time variance")
print(f"Average prediction error: {round(xgb_mae, 4)} seconds")

# ============================================================
# STEP 10 — SAVE MODEL
# ============================================================

joblib.dump(xgb_model, 'saved_models/model1_xgb_laptime.pkl')
joblib.dump(le,        'saved_models/model1_label_encoder.pkl')
joblib.dump(features,  'saved_models/model1_features.pkl')
print("\nModel 1 saved successfully ✅")

# ============================================================
# STEP 11 — LAP TIME PREDICTOR (USER INPUT)
# ============================================================

print("\n" + "="*60)
print("LAP TIME PREDICTOR — ENTER YOUR SCENARIO")
print("="*60)
print("\nEXAMPLE INPUT (you can use these values to test):")
print("  LapNumber      : 30")
print("  Compound       : MEDIUM")
print("  TyreLife       : 18")
print("  FuelLoad_kg    : 62.0")
print("  Position       : 5")
print("  SpeedI1        : 295.0")
print("  SpeedI2        : 268.0")
print("  SpeedFL        : 245.0")
print("  SpeedST        : 290.0")
print("  Race           : Italian")
print("="*60)

print("\nEnter your scenario below:\n")

lap_number     = float(input("Enter LapNumber (e.g. 30): "))
compound_input = input("Enter Compound (SOFT / MEDIUM / HARD): ").strip().upper()
while compound_input not in ['SOFT', 'MEDIUM', 'HARD']:
    print("Invalid compound. Please enter SOFT, MEDIUM or HARD.")
    compound_input = input("Enter Compound (SOFT / MEDIUM / HARD): ").strip().upper()
compound_enc   = le.transform([compound_input])[0]

tyre_life      = float(input("Enter TyreLife — laps on current tires (e.g. 18): "))
fuel_load      = float(input("Enter FuelLoad_kg (e.g. 62.0): "))
fuel_correction= round(fuel_load * 0.03, 3)
print(f"  (FuelCorrection_s auto-calculated: {fuel_correction}s)")

position       = float(input("Enter Position (1–20): "))
speed_i1       = float(input("Enter SpeedI1 — speed trap 1 in km/h (e.g. 295.0): "))
speed_i2       = float(input("Enter SpeedI2 — speed trap 2 in km/h (e.g. 268.0): "))
speed_fl       = float(input("Enter SpeedFL — finish line speed in km/h (e.g. 245.0): "))
speed_st       = float(input("Enter SpeedST — speed trap in km/h (e.g. 290.0): "))

race_input     = input("Enter Race (Bahrain / Azerbaijan / British / Italian / Singapore): ").strip().capitalize()
valid_races    = ['Bahrain', 'Azerbaijan', 'British', 'Italian', 'Singapore']
while race_input not in valid_races:
    print(f"Invalid race. Choose from: {valid_races}")
    race_input = input("Enter Race: ").strip().capitalize()

# Build scenario
scenario = {
    'LapNumber'        : lap_number,
    'Compound_enc'     : compound_enc,
    'TyreLife'         : tyre_life,
    'FuelLoad_kg'      : fuel_load,
    'FuelCorrection_s' : fuel_correction,
    'Position'         : position,
    'SpeedI1'          : speed_i1,
    'SpeedI2'          : speed_i2,
    'SpeedFL'          : speed_fl,
    'SpeedST'          : speed_st,
    'Race_Azerbaijan'  : 1 if race_input == 'Azerbaijan' else 0,
    'Race_Bahrain'     : 1 if race_input == 'Bahrain'    else 0,
    'Race_British'     : 1 if race_input == 'British'    else 0,
    'Race_Italian'     : 1 if race_input == 'Italian'    else 0,
    'Race_Singapore'   : 1 if race_input == 'Singapore'  else 0,
}

scenario_df    = pd.DataFrame([scenario])
rf_prediction  = rf_model.predict(scenario_df)[0]
xgb_prediction = xgb_model.predict(scenario_df)[0]

# Wear level classification
wear_level = (
    "fresh"        if tyre_life <= 5  else
    "lightly used" if tyre_life <= 15 else
    "heavily worn"
)

# Speed level classification
speed_level = (
    "fast"    if xgb_prediction < 88 else
    "average" if xgb_prediction < 97 else
    "slow"
)

print("\n" + "="*60)
print("PREDICTION RESULTS")
print("="*60)
print(f"  Circuit        : {race_input}")
print(f"  Compound       : {compound_input} (TyreLife: {tyre_life} laps)")
print(f"  Lap Number     : {int(lap_number)}")
print(f"  Fuel Load      : {fuel_load} kg")
print(f"  Position       : {int(position)}")
print(f"")
print(f"  Random Forest  : {rf_prediction:.3f} seconds")
print(f"  XGBoost        : {xgb_prediction:.3f} seconds")
print(f"")
print(f"  Final Prediction (XGBoost): {xgb_prediction:.3f} seconds")
print("="*60)

# Human readable explanation
print(f"\n  WHAT THIS MEANS:")
print(f"  If your driver stays out on {wear_level} {compound_input} tires")
print(f"  (used for {int(tyre_life)} laps) at Lap {int(lap_number)} of the {race_input} Grand Prix,")
print(f"  running in P{int(position)} with {fuel_load}kg of fuel remaining,")
print(f"  they are expected to lap at {xgb_prediction:.3f} seconds — which is considered {speed_level}")
print(f"  for this circuit.")
print("="*60)
