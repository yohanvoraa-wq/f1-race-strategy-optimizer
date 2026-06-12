# ============================================================
# UNIFIED F1 ML PIPELINE
# Model 1 — Lap Time Prediction
# Model 2 — Race Strategy Optimization
# Dynamic Programming — Breakeven Analysis
# Majority Vote — Final Decision
# F1 ML Project | NMIMS STME | Semester 4
# Team: Yohan Vora (A186) | Jugraj Singh (A188)
# Submitted To: Dr. Divyang Jadav
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1 — LOAD SAVED MODELS
# ============================================================

print("="*60)
print("UNIFIED F1 RACE STRATEGY PIPELINE")
print("="*60)

xgb_laptime  = joblib.load('saved_models/model1_xgb_laptime.pkl')
le_m1        = joblib.load('saved_models/model1_label_encoder.pkl')
features_m1  = joblib.load('saved_models/model1_features.pkl')

xgb_strategy = joblib.load('saved_models/model2_xgb_strategy.pkl')
le_m2        = joblib.load('saved_models/model2_label_encoder.pkl')
features_m2  = joblib.load('saved_models/model2_features.pkl')

print("Model 1 — Lap Time Prediction    : LOADED ✅")
print("Model 2 — Race Strategy Optimizer: LOADED ✅")

# ============================================================
# STEP 2 — PIPELINE CONSTANTS
# ============================================================

pit_loss_times = {
    'Azerbaijan' : 24.0,
    'Bahrain'    : 22.0,
    'British'    : 20.0,
    'Italian'    : 21.0,
    'Singapore'  : 28.0
}

total_laps = {
    'Azerbaijan' : 51,
    'Bahrain'    : 57,
    'British'    : 52,
    'Italian'    : 53,
    'Singapore'  : 62
}

races = ['Bahrain', 'Azerbaijan', 'British', 'Italian', 'Singapore']

# ============================================================
# STEP 3 — HELPER FUNCTIONS
# ============================================================

def build_laptime_input(lap_number, compound, tyre_life,
                        fuel_load, position,
                        speed_i1, speed_i2, speed_fl, speed_st,
                        race, le):
    compound_enc    = le.transform([compound])[0]
    fuel_correction = round(fuel_load * 0.03, 3)
    row = {
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
        'Race_Azerbaijan'  : 1 if race == 'Azerbaijan' else 0,
        'Race_Bahrain'     : 1 if race == 'Bahrain'    else 0,
        'Race_British'     : 1 if race == 'British'    else 0,
        'Race_Italian'     : 1 if race == 'Italian'    else 0,
        'Race_Singapore'   : 1 if race == 'Singapore'  else 0,
    }
    return pd.DataFrame([row])

def build_strategy_input(lap_number, compound, tyre_life,
                         fuel_load, deg_rate, laps_remaining,
                         position, safety_car, red_flag,
                         race, le):
    compound_enc    = le.transform([compound])[0]
    fuel_correction = round(fuel_load * 0.03, 3)
    row = {
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
        'Race_Azerbaijan'  : 1 if race == 'Azerbaijan' else 0,
        'Race_Bahrain'     : 1 if race == 'Bahrain'    else 0,
        'Race_British'     : 1 if race == 'British'    else 0,
        'Race_Italian'     : 1 if race == 'Italian'    else 0,
        'Race_Singapore'   : 1 if race == 'Singapore'  else 0,
    }
    return pd.DataFrame([row])

def breakeven_check(deg_rate, laps_remaining, pit_loss):
    if deg_rate <= 0:
        return None
    time_gained    = deg_rate * laps_remaining
    breakeven_laps = pit_loss / deg_rate
    decision       = "PIT" if time_gained > pit_loss else "STAY OUT"
    return time_gained, breakeven_laps, decision

# ============================================================
# STEP 4 — USER INPUT
# ============================================================

print("\n" + "="*60)
print("ENTER RACE SCENARIO")
print("="*60)
print("\nEXAMPLE INPUT (PIT scenario):")
print("  Race          : Bahrain")
print("  LapNumber     : 15")
print("  Compound      : SOFT")
print("  TyreLife      : 25")
print("  FuelLoad_kg   : 80.0")
print("  DegRate_s_lap : 0.60")
print("  LapsRemaining : 42")
print("  Position      : 3")
print("  SafetyCar     : 1")
print("  RedFlag       : 0")
print("  SpeedI1       : 295.0")
print("  SpeedI2       : 268.0")
print("  SpeedFL       : 245.0")
print("  SpeedST       : 290.0")
print("="*60)

print("\nEnter your scenario:\n")

race_input = input("Enter Race (Bahrain/Azerbaijan/British/Italian/Singapore): ").strip().capitalize()
while race_input not in races:
    race_input = input(f"Invalid. Choose from {races}: ").strip().capitalize()

lap_number  = float(input("Enter LapNumber: "))
compound    = input("Enter Compound (SOFT/MEDIUM/HARD): ").strip().upper()
while compound not in ['SOFT', 'MEDIUM', 'HARD']:
    compound = input("Invalid. Enter SOFT/MEDIUM/HARD: ").strip().upper()
tyre_life      = float(input("Enter TyreLife (laps on current tires): "))
fuel_load      = float(input("Enter FuelLoad_kg: "))
deg_rate       = float(input("Enter DegRate_s_lap (e.g. 0.60): "))
laps_remaining = float(input("Enter LapsRemaining: "))
position       = float(input("Enter Position (1-20): "))
safety_car     = int(input("SafetyCar active? (0=No / 1=Yes): "))
red_flag       = int(input("RedFlag active? (0=No / 1=Yes): "))
speed_i1       = float(input("Enter SpeedI1 km/h (e.g. 295.0): "))
speed_i2       = float(input("Enter SpeedI2 km/h (e.g. 268.0): "))
speed_fl       = float(input("Enter SpeedFL km/h (e.g. 245.0): "))
speed_st       = float(input("Enter SpeedST km/h (e.g. 290.0): "))

# ============================================================
# STEP 5 — RUN PIPELINE
# ============================================================

pit_loss = pit_loss_times[race_input]

# Model 1 — Current lap time on worn tires
m1_current      = build_laptime_input(
    lap_number, compound, tyre_life,
    fuel_load, position,
    speed_i1, speed_i2, speed_fl, speed_st,
    race_input, le_m1
)
current_laptime = xgb_laptime.predict(m1_current)[0]

# Model 1 — Predicted lap time on fresh SOFT tires after pit
m1_fresh        = build_laptime_input(
    lap_number + 1, 'SOFT', 1,
    fuel_load - 1.6, position,
    speed_i1, speed_i2, speed_fl, speed_st,
    race_input, le_m1
)
fresh_laptime   = xgb_laptime.predict(m1_fresh)[0]

# Model 2 — Strategy decision
m2_input          = build_strategy_input(
    lap_number, compound, tyre_life,
    fuel_load, deg_rate, laps_remaining,
    position, safety_car, red_flag,
    race_input, le_m2
)
strategy_decision = xgb_strategy.predict(m2_input)[0]
strategy_label    = "PIT" if strategy_decision == 1 else "STAY OUT"

# Dynamic Programming Breakeven
dp_result = breakeven_check(deg_rate, laps_remaining, pit_loss)

# ============================================================
# STEP 6 — MAJORITY VOTE ENGINE
# ============================================================

# Three signals vote:
# Vote 1 — Model 1: Is lap time gain per lap > 0.5s on fresh tires?
lap_vote = 1 if (current_laptime - fresh_laptime) > 0.5 else 0

# Vote 2 — Model 2: XGBoost classification
ml_vote  = int(strategy_decision)

# Vote 3 — Dynamic Programming: Time Gained > Pit Loss
dp_vote  = 1 if dp_result and dp_result[2] == "PIT" else 0

votes          = lap_vote + ml_vote + dp_vote
final_decision = "PIT 🔴" if votes >= 2 else "STAY OUT 🟢"

# ============================================================
# STEP 7 — DISPLAY UNIFIED OUTPUT
# ============================================================

wear_level = (
    "Fresh"        if tyre_life <= 5  else
    "Lightly Used" if tyre_life <= 15 else
    "Heavily Worn"
)

print("\n" + "="*60)
print("UNIFIED PIPELINE — RACE STRATEGY REPORT")
print("="*60)
print(f"  Circuit           : {race_input} Grand Prix")
print(f"  Lap               : {int(lap_number)} of {total_laps[race_input]}")
print(f"  Current Position  : P{int(position)}")
print(f"  Safety Car        : {'Yes ⚠️' if safety_car else 'No'}")
print(f"  Red Flag          : {'Yes 🚩' if red_flag   else 'No'}")
print()
print(f"  --- TIRE STATUS ---")
print(f"  Compound          : {compound} (Age: {int(tyre_life)} laps)")
print(f"  Degradation Rate  : {deg_rate} s/lap")
print(f"  Tire Condition    : {wear_level}")
print()
print(f"  --- MODEL 1: LAP TIME PREDICTION ---")
print(f"  Current Lap Time  : {current_laptime:.3f}s  (on {compound}, {int(tyre_life)} laps old)")
print(f"  Fresh Tire Lap    : {fresh_laptime:.3f}s  (on fresh SOFT after pit)")
print(f"  Lap Time Gain     : {current_laptime - fresh_laptime:.3f}s per lap on fresh tires")
print()
print(f"  --- MODEL 2: STRATEGY DECISION ---")
print(f"  XGBoost Decision  : {strategy_label}")
print()
print(f"  --- DYNAMIC PROGRAMMING ANALYSIS ---")
if dp_result:
    time_gained, be_laps, dp_rec = dp_result
    print(f"  Pit Loss Time     : {pit_loss}s")
    print(f"  Time Gained       : {time_gained:.2f}s over {int(laps_remaining)} laps")
    print(f"  Breakeven Point   : {be_laps:.1f} laps on fresh tires")
    print(f"  DP Recommendation : {dp_rec}")
print()
print(f"  --- FINAL UNIFIED RECOMMENDATION ---")
print(f"  Model 1 Vote      : {'PIT' if lap_vote else 'STAY OUT'}")
print(f"  Model 2 Vote      : {strategy_label}")
print(f"  DP Vote           : {dp_result[2] if dp_result else 'STAY OUT'}")
print(f"  Total PIT Votes   : {votes}/3")
print(f"  FINAL DECISION    : {final_decision}")
print()

# Human readable summary
sc_note = " A Safety Car is on track — this is the perfect free pit window." if safety_car else ""
print(f"  WHAT THIS MEANS:")
print(f"  Your driver is on {wear_level} {compound} tires at Lap {int(lap_number)}")
print(f"  of the {race_input} GP with {int(laps_remaining)} laps remaining.{sc_note}")
print(f"  Pitting now would cost {pit_loss}s but gain")
print(f"  {current_laptime - fresh_laptime:.3f}s per lap on fresh tires.")
print(f"  Pipeline recommends: {final_decision}")
print("="*60)
