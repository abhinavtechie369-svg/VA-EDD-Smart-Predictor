# VA-EDD Smart Predictor

## Overview

VA-EDD Smart Predictor is a Machine Learning-based decision support system developed for analyzing and optimizing the Vibration Assisted Electric Discharge Drilling (VA-EDD) process.

The system predicts key machining performance indicators:

* Material Removal Rate (MRR)
* Surface Roughness (SR)

using process parameters such as Peak Current (IP), Pulse ON Time (TON), Pulse OFF Time (TOFF), Tool Rotation (TR), and Vibration Amplitude (VA).

An interactive Streamlit dashboard enables engineers and researchers to visualize parameter relationships, compare machine learning models, perform simulations, and identify optimal machining conditions.

---

## Features

### Machine Learning Prediction

* Multi-output prediction of MRR and SR
* Ridge Regression model
* Random Forest Regression model
* Feature engineering using interaction terms

### Data Analysis

* Exploratory Data Analysis (EDA)
* Correlation Heatmaps
* Parameter Relationship Analysis

### Interactive Dashboard

* Real-time prediction interface
* Adjustable machining parameters
* Model switching (Ridge / Random Forest)

### Advanced Visualization

* Interactive 2D plots
* 3D surface visualization
* 4D parameter visualization
* Correlation analysis

### Optimization

* Genetic Algorithm based parameter optimization
* Trade-off analysis between MRR and SR

### Simulation

* Dynamic simulation of parameter effects
* Visualization of machining behavior under changing conditions

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Plotly
* SciPy

---

## Project Structure

```text
VA_EDD_ML/
│
├── app.py
├── eda_analysis.py
├── model_training.py
├── requirements.txt
├── README.md
│
├── data/
│   └── va_edd_dataset.csv
│
└── screenshots/
```

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

---

## Future Improvements

* Deep Learning models
* Multi-objective optimization
* Digital Twin integration
* Industrial IoT connectivity
* Real-time sensor-based prediction

---

## Author

**Abhinav Mishra**
B.Tech Mechanical Engineering
University Institute of Engineering & Technology (UIET), CSJMU Kanpur
