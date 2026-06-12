# Formula 1 Race Strategy Optimizer

### Machine Learning Powered Pit Stop Decision System Using FastF1 Telemetry and Predictive Analytics

---

## Project Overview

Formula 1 race strategy is one of the most influential factors in determining race outcomes. Teams continuously analyze tire degradation, fuel load, weather conditions, driver performance, and pit stop timing to maximize track position and overall race pace.

This project presents an end-to-end machine learning based race strategy optimization system that leverages real Formula 1 telemetry data collected through the FastF1 API. The system predicts lap performance, evaluates pit stop opportunities, and generates race strategy recommendations through a combination of predictive modeling and optimization techniques.

The solution was developed as part of a Machine Learning course project for the B.Tech Artificial Intelligence and Data Science program at SVKM's NMIMS STME and demonstrates the practical application of machine learning within motorsport analytics.

---

## Problem Statement

Race engineers must constantly answer a critical question during every race:

**Should the driver pit now or stay out for another lap?**

Making the wrong decision can result in:

* Significant time loss
* Increased tire degradation
* Reduced race pace
* Loss of track position
* Missed strategic opportunities

The objective of this project is to develop a data-driven decision support system capable of recommending optimal pit stop timing using historical Formula 1 race data and machine learning models.

---

## Dataset

### Data Source

FastF1 Python API (Official Formula 1 Timing and Telemetry Data)

### Season

2023 Formula 1 World Championship

### Circuits Analyzed

* Bahrain Grand Prix
* Azerbaijan Grand Prix
* British Grand Prix (Silverstone)
* Italian Grand Prix (Monza)
* Singapore Grand Prix

### Available Data Per Circuit

Each circuit dataset contains:

* Lap-by-lap timing data
* Tire strategy information
* Driver performance statistics
* Weather conditions
* Telemetry measurements

### Feature Engineering

The raw race data was transformed into machine learning features including:

* Fuel Load Estimation
* Fuel Time Correction
* Tire Age
* Tire Degradation Rate
* Compound Type Encoding
* Pit Stop Loss Estimation
* Driver Performance Metrics
* Weather Features
* Safety Car Conditions
* Historical Lap Trends

---

## System Architecture

FASTF1 DATA (5 Circuits)

Bahrain
Azerbaijan
Silverstone
Monza
Singapore

↓

Feature Engineering

Fuel Load
Fuel Correction
Tyre Age
Tyre Degradation Rate
Pit Loss Time
Weather Features
Safety Car Flags

↓

Model 1 – Lap Time Prediction

(XGBoost Regressor)

↓

Model 2 – Pit Stop Decision Classification

(XGBoost Classifier)

↓

Dynamic Programming

Breakeven Analysis

↓

Majority Vote Decision Engine

↓

FINAL STRATEGY

PIT or STAY OUT

---

## Model 1 – Lap Time Prediction

### Objective

Predict future lap performance based on race conditions and tire state.

### Algorithm

XGBoost Regressor

### Target Variable

LapTime_s

### Performance

| Metric   | Value    |
| -------- | -------- |
| R² Score | 0.9899   |
| MAE      | 0.3211 s |
| RMSE     | 0.6978 s |

### Purpose

The model estimates:

* Current lap performance
* Future tire degradation effects
* Expected pace after a pit stop
* Fresh tire performance gains

---

## Model 2 – Pit Strategy Classification

### Objective

Determine whether a driver should pit or remain on track.

### Algorithm

XGBoost Classifier

### Target Variable

IsPitLap

* 0 = Stay Out
* 1 = Pit

### Performance

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 92.94% |
| Recall   | 50.00% |
| F1 Score | 30.00% |

### Purpose

The model learns strategic patterns from historical races and predicts whether current race conditions justify a pit stop.

---

## Dynamic Programming Optimization

Machine learning predictions alone are not sufficient for race strategy optimization.

To improve decision quality, a Dynamic Programming based breakeven analysis module evaluates:

* Time lost during pit stops
* Time gained through fresh tires
* Remaining race distance
* Tire degradation impact
* Future lap performance

This enables the system to quantify whether a pit stop can realistically recover the time lost in the pit lane.

---

## Unified Decision Engine

The final recommendation is generated through a voting mechanism combining:

1. Lap Time Prediction Model
2. Pit Strategy Classification Model
3. Dynamic Programming Breakeven Analysis

The majority decision produces the final recommendation:

🟢 STAY OUT

or

🔴 PIT NOW

This approach reduces dependence on any single model and improves overall robustness.

---

## Technologies Used

### Machine Learning

* XGBoost
* Scikit-learn
* Pandas
* NumPy

### Data Collection

* FastF1 API

### Data Processing

* Feature Engineering
* Data Cleaning
* Statistical Analysis

### Optimization

* Dynamic Programming
* Majority Vote Ensemble Logic

### Programming Language

* Python

---

## Project Structure

```text
f1-race-strategy-optimizer/
│
├── data/
│   ├── bahrain/
│   ├── azerbaijan/
│   ├── silverstone/
│   ├── monza/
│   └── singapore/
│
├── src/
│   ├── lap_time_prediction.py
│   ├── pit_strategy_classifier.py
│   └── unified_strategy_pipeline.py
│
├── models/
│
├── notebooks/
│
├── docs/
│
├── presentation/
│
├── requirements.txt
│
└── README.md
```

---

## Key Contributions

* Built a machine learning based race strategy optimization framework.
* Processed and engineered features from real Formula 1 telemetry data.
* Developed an XGBoost regression model for lap time prediction.
* Developed an XGBoost classification model for pit stop recommendations.
* Implemented Dynamic Programming based breakeven analysis.
* Combined multiple decision sources through an ensemble voting framework.
* Evaluated performance across five Formula 1 Grand Prix circuits.

---

## Future Improvements

* Reinforcement Learning based race strategy optimization
* Real-time race simulation engine
* Safety Car and Virtual Safety Car modeling
* Multi-driver race strategy planning
* Weather forecasting integration
* Deep Learning based lap prediction models

---

## Authors

**Yohan Vora**

B.Tech Artificial Intelligence & Data Science

SVKM's NMIMS Mukesh Patel School of Technology Management & Engineering

**Jugraj Singh**

B.Tech Artificial Intelligence & Data Science

SVKM's NMIMS Mukesh Patel School of Technology Management & Engineering

---

## Acknowledgements

* FastF1 Development Team
* Formula 1 Official Timing Data
* SVKM's NMIMS STME
