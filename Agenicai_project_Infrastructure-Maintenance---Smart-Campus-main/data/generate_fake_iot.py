import argparse
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Generator, List

import numpy as np
import pandas as pd
from pathlib import Path


ASSET_TYPES = [
	"HVAC",
	"Chiller",
	"Boiler",
	"Elevator",
	"Lighting",
	"Pump",
]


def _seed_everything(seed: int) -> None:
	"""Seed RNGs for reproducible generation."""
	random.seed(seed)
	np.random.seed(seed)


def generate_assets(num_assets: int, campus: str = "Main") -> pd.DataFrame:
	"""Create a catalog of assets with static metadata."""
	assets = []
	for asset_id in range(1, num_assets + 1):
		asset_type = random.choice(ASSET_TYPES)
		building = f"Building-{random.randint(1, 6)}"
		age_years = np.clip(np.random.normal(7, 3), 0.5, 25)
		assets.append(
			{
				"asset_id": f"A{asset_id:03d}",
				"asset_type": asset_type,
				"building": building,
				"campus": campus,
				"age_years": round(float(age_years), 2),
			}
		)
	return pd.DataFrame(assets)


def generate_time_series(
	num_assets: int,
	hours: int,
	start: datetime | None = None,
	frequency_minutes: int = 10,
	seed: int = 42,
) -> pd.DataFrame:
	"""Generate synthetic IoT sensor readings across time for multiple assets."""
	_seed_everything(seed)
	assets_df = generate_assets(num_assets)
	start_ts = start or (datetime.utcnow() - timedelta(hours=hours))
	periods = (hours * 60) // frequency_minutes

	rows: List[Dict] = []
	for _, asset in assets_df.iterrows():
		base_temp = 22 + np.random.normal(0, 1)
		base_humidity = 45 + np.random.normal(0, 3)
		base_vibration = 0.3 + abs(np.random.normal(0, 0.1))
		base_power_kw = 5 + abs(np.random.normal(0, 1.5))
		usage_hours_daily = np.clip(np.random.normal(10, 3), 2, 20)

		for i in range(int(periods)):
			ts = start_ts + timedelta(minutes=frequency_minutes * i)
			temp = base_temp + np.sin(i / 12) * 2 + np.random.normal(0, 0.5)
			humidity = base_humidity + np.sin(i / 18) * 5 + np.random.normal(0, 1)
			vibration = base_vibration + np.random.normal(0, 0.03)
			power_kw = base_power_kw + (usage_hours_daily / 10) + np.random.normal(0, 0.5)

			# Inject occasional anomalies
			if random.random() < 0.01:
				temp += np.random.uniform(5, 10)
				vibration += np.random.uniform(0.3, 0.6)
				power_kw += np.random.uniform(3, 8)

			rows.append(
				{
					"timestamp": ts,
					"asset_id": asset["asset_id"],
					"asset_type": asset["asset_type"],
					"building": asset["building"],
					"campus": asset["campus"],
					"age_years": asset["age_years"],
					"temperature_c": round(float(temp), 2),
					"humidity_pct": round(float(humidity), 2),
					"vibration_g": round(float(vibration), 3),
					"power_kw": round(float(power_kw), 2),
					"usage_hours": round(float(usage_hours_daily), 2),
				}
			)

	df = pd.DataFrame(rows)
	# Create a proxy failure label: high temp/vibration/power and old age -> higher risk
	score = (
		(df["temperature_c"] - df["temperature_c"].rolling(12, min_periods=1).mean()).clip(lower=0)
		+ (df["vibration_g"] - df["vibration_g"].rolling(12, min_periods=1).mean()).clip(lower=0) * 5
		+ (df["power_kw"] - df["power_kw"].rolling(12, min_periods=1).mean()).clip(lower=0)
		+ (df["age_years"] / 25)
	)
	prob = 1 / (1 + np.exp(-(score - score.mean()) / (score.std() + 1e-6)))
	df["failure_within_30d"] = (np.random.rand(len(df)) < prob * 0.3).astype(int)
	return df


def stream_readings(df: pd.DataFrame, delay_secs: float = 0.5) -> Generator[Dict, None, None]:
	"""Yield rows as simulated live stream with periodic delay."""
	for _, row in df.sort_values("timestamp").iterrows():
		yield row.to_dict()
		time.sleep(delay_secs)


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate synthetic IoT readings")
	parser.add_argument("--output", type=str, default="data/iot_readings.csv", help="CSV output path")
	parser.add_argument("--assets", type=int, default=20, help="Number of assets to simulate")
	parser.add_argument("--hours", type=int, default=48, help="How many hours of history to generate")
	parser.add_argument("--freq", type=int, default=10, help="Sampling frequency in minutes")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	df = generate_time_series(
		num_assets=args.assets,
		hours=args.hours,
		frequency_minutes=args.freq,
		seed=args.seed,
	)

	output_path = args.output
	# Ensure directory exists
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_path, index=False)
	print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
	main()

