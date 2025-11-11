import os
import subprocess
from pathlib import Path

DATA_CSV = Path("data/iot_readings.csv")
MODEL = Path("models/model.joblib")

DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
MODEL.parent.mkdir(parents=True, exist_ok=True)

if not DATA_CSV.exists():
	print("[startup] Generating sample IoT data...")
	subprocess.check_call([
		"python", "data/generate_fake_iot.py", "--output", str(DATA_CSV), "--assets", "25", "--hours", "72"
	])

if not MODEL.exists():
	print("[startup] Training predictive model...")
	subprocess.check_call([
		"python", "ml/train_predictive_model.py", "--input", str(DATA_CSV), "--model", str(MODEL)
	])

port = os.environ.get("PORT", os.environ.get("STREAMLIT_SERVER_PORT", "8080"))
subprocess.check_call([
	"streamlit", "run", "app/streamlit_app.py", "--server.port=" + str(port), "--server.address=0.0.0.0"
])
