# AI-Powered Digital Twin for Traffic Safety Risk Prediction

## Project Overview
This project simulates a **Cyber-Physical System (CPS)** for an intelligent transportation network. It establishes a **Digital Twin**—a cyber representation of a physical traffic intersection or highway segment—to monitor, predict, and mitigate traffic safety risks in real-time.

By bridging the physical layer (simulated vehicles, weather, traffic flow) with a cyber layer (AI prediction models, LLM-based reasoning, Streamlit dashboard), the system provides an end-to-end framework for proactive traffic management.

## System Architecture

The architecture illustrates the bi-directional flow between the physical and cyber components:

### 1. Physical Layer (Data Generation & Sensing)
Represents the real world. Data collected includes:
* **Vehicle Speed:** Average flow speed.
* **Traffic Flow:** Number of vehicles per hour.
* **Lane Occupancy:** Percentage of road utilized.
* **Environment:** Weather conditions (Clear, Rain, Fog, Snow).
* **Driver Behavior:** Sudden braking events.

### 2. Cyber Layer (Digital Twin & Analytics)
The digital replica that processes data and runs models:
* **Machine Learning Engine:** 
  * *Random Forest Regressors* predict Collision Probability and Congestion Level.
  * *Random Forest Classifier* predicts the Risk Category (Low, Medium, High).
  * *Isolation Forest* detects anomalous traffic patterns.
* **LLM Reasoning Module:** Analyzes the structured output from the ML models to generate human-readable incident analyses, root-cause derivations, and mitigation strategies using Large Language Models (Gemini integration ready).
* **Actuation / Recommendation:** Generates physical interventions (e.g., modifying traffic signal timings) based on cyber analysis.

### 3. User Interface (Streamlit Dashboard)
A professional, research-oriented dashboard visualizing the Digital Twin state, KPIs, model predictions, and LLM outputs in real-time.

## Key Features & Research Contributions
* **Cyber-Physical Representation:** Accurately models the feedback loop of a smart city traffic system.
* **Multi-Model Predictive AI:** Combines regression, classification, and anomaly detection for holistic safety monitoring.
* **LLM Integration for Analytics:** Moves beyond simple alerting to automated, descriptive incident reasoning.
* **Synthetic Data Generation:** Includes a robust script modeling complex physical interactions (e.g., weather impact on speed and occupancy) to train the models.

## Project Structure
```text
├── data_generator.py      # Generates synthetic CPS traffic data
├── model_training.py      # Trains Scikit-learn predictive & anomaly models
├── llm_analysis.py        # Handles GenAI/LLM incident reporting 
├── utils.py               # Helper functions for feature engineering
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── models/                # Saved trained models (.pkl)
└── README.md              # Project documentation
```

## Setup & Deployment Instructions

### 1. Prerequisites
* Python 3.9+
* Recommended: Virtual environment (venv or conda)

### 2. Installation
Clone the repository or navigate to the project directory, then install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. LLM Setup (Optional but Recommended)
To use the live Google Gemini API for incident analysis:
1. Obtain an API key from Google AI Studio.
2. Create a `.env` file in the root directory.
3. Add your key: `GEMINI_API_KEY=your_api_key_here`
*Note: If no API key is provided, the system falls back to an internal mock LLM response system that mimics the required analytical structure.*

### 4. Running the Application
Run the Streamlit application using the following command:
```bash
streamlit run app.py
```
*Note: Upon the first run, the app will automatically generate the synthetic dataset (`synthetic_traffic_data.csv`) and train the required models. This may take a few moments.*

## Future Research Extensions
* **V2X Integration:** Simulate Vehicle-to-Everything communication protocols feeding live micro-data into the twin.
* **Deep Learning for Trajectory Prediction:** Implement LSTM or Spatio-Temporal Graph Convolutional Networks (STGCN) for predicting exact vehicle paths.
* **Reinforcement Learning:** Train an RL agent to control the traffic signals dynamically based on the digital twin's state reward function.
