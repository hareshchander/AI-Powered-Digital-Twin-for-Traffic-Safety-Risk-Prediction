import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from utils import prepare_features

def train_and_save_models(data_path="synthetic_traffic_data.csv", models_dir="models"):
    """
    Trains ML models for prediction and anomaly detection and saves them.
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print("Preparing features...")
    X = prepare_features(df)
    
    y_collision = df['collision_probability']
    y_congestion = df['congestion_level']
    y_risk = df['risk_category']

    # Splitting data
    X_train, X_test, y_coll_train, y_coll_test, y_cong_train, y_cong_test, y_risk_train, y_risk_test = train_test_split(
        X, y_collision, y_congestion, y_risk, test_size=0.2, random_state=42
    )

    print("Training Collision Probability Model...")
    rf_collision = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_collision.fit(X_train, y_coll_train)
    coll_preds = rf_collision.predict(X_test)
    print(f"Collision Model MSE: {mean_squared_error(y_coll_test, coll_preds):.4f}")
    joblib.dump(rf_collision, os.path.join(models_dir, 'collision_model.pkl'))

    print("Training Congestion Level Model...")
    rf_congestion = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_congestion.fit(X_train, y_cong_train)
    cong_preds = rf_congestion.predict(X_test)
    print(f"Congestion Model MSE: {mean_squared_error(y_cong_test, cong_preds):.4f}")
    joblib.dump(rf_congestion, os.path.join(models_dir, 'congestion_model.pkl'))

    print("Training Risk Category Classifier...")
    rf_risk = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_risk.fit(X_train, y_risk_train)
    risk_preds = rf_risk.predict(X_test)
    print(f"Risk Model Accuracy: {accuracy_score(y_risk_test, risk_preds):.4f}")
    print(classification_report(y_risk_test, risk_preds))
    joblib.dump(rf_risk, os.path.join(models_dir, 'risk_model.pkl'))

    print("Training Anomaly Detection Model...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X)
    joblib.dump(iso_forest, os.path.join(models_dir, 'anomaly_model.pkl'))

    print("All models trained and saved successfully.")

if __name__ == "__main__":
    train_and_save_models()
