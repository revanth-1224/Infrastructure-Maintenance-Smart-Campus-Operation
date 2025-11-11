import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS: List[str] = [
	"temperature_c",
	"humidity_pct",
	"vibration_g",
	"power_kw",
	"usage_hours",
	"age_years",
]

TARGET_COLUMN = "failure_within_30d"


def load_dataset(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path, parse_dates=["timestamp"])  # timestamp may be useful later
	# Minimal cleaning: drop rows with missing required values
	return df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
	x = df[FEATURE_COLUMNS].values
	y = df[TARGET_COLUMN].values
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

	model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
	model.fit(x_train, y_train)

	y_pred = model.predict(x_test)
	print(classification_report(y_test, y_pred))
	return model


def main() -> None:
	parser = argparse.ArgumentParser(description="Train predictive maintenance model")
	parser.add_argument("--input", type=str, default="data/iot_readings.csv", help="Input CSV")
	parser.add_argument("--model", type=str, default="models/model.joblib", help="Output model path")
	args = parser.parse_args()

	df = load_dataset(args.input)
	model = train_model(df)

	model_path = Path(args.model)
	model_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, model_path)
	print(f"Model saved to {model_path}")


if __name__ == "__main__":
	main()

