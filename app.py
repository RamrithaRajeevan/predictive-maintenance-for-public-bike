from flask import Flask, render_template
import pandas as pd
import joblib
import random

app = Flask(__name__)

# Load model
model = joblib.load("xgboost_bike_failure_model.pkl")

# Load dataset
df = pd.read_csv("bike_component_data.csv")

@app.route("/")
def home():
    # Add random risks for demo (you can replace with model.predict if needed)
    df["brake_risk"] = [random.randint(10, 95) for _ in range(len(df))]
    df["chain_risk"] = [random.randint(10, 95) for _ in range(len(df))]
    df["gear_risk"] = [random.randint(10, 95) for _ in range(len(df))]
    df["tire_risk"] = [random.randint(10, 95) for _ in range(len(df))]

    # Calculate overall risk
    df["overall_risk"] = df[["brake_risk", "chain_risk", "gear_risk", "tire_risk"]].mean(axis=1)

    # Classify bikes
    df["status"] = df["overall_risk"].apply(lambda r: "Needs Maintenance" if r > 60 else "Healthy")
    df["color"] = df["status"].apply(lambda s: "red" if s == "Needs Maintenance" else "green")

    # Select 3 healthy and 3 needing maintenance
    maintenance_bikes = df[df["status"] == "Needs Maintenance"].sample(n=3, random_state=42)
    healthy_bikes = df[df["status"] == "Healthy"].sample(n=3, random_state=21)

    selected_bikes = pd.concat([maintenance_bikes, healthy_bikes])

    bikes = []
    for _, row in selected_bikes.iterrows():
        bikes.append({
            "id": f"Bike-{random.randint(1000,9999)}",
            "distance": row["distance_travelled_km"],
            "brake_risk": int(row["brake_risk"]),
            "chain_risk": int(row["chain_risk"]),
            "gear_risk": int(row["gear_risk"]),
            "tire_risk": int(row["tire_risk"]),
            "status": row["status"],
            "color": row["color"]
        })

    total_bikes = len(df)
    needs_maintenance = len(df[df["status"] == "Needs Maintenance"])
    healthy = total_bikes - needs_maintenance

    return render_template("index.html", bikes=bikes, total_bikes=total_bikes,
                           healthy=healthy, needs_maintenance=needs_maintenance)

if __name__ == "__main__":
    app.run(debug=True)
