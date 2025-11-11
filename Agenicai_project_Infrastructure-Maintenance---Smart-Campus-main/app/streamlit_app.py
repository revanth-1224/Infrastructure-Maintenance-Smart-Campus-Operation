import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
	from dotenv import load_dotenv  # type: ignore
	load_dotenv()
except Exception:
	pass

# Ensure project root is importable
import sys as _sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
	_sys.path.insert(0, str(_PROJECT_ROOT))

from agents.langgraph_graph import run_workflow

CSV_PATH = "data/iot_readings.csv"
MODEL_PATH = "models/model.joblib"

st.set_page_config(page_title="Smart Campus", layout="wide")
st.title("🏫 Smart Campus")
st.caption("Simple controls. Clear info. Cool look.")


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
	if not Path(path).exists():
		st.warning("We couldn't find your data yet. Please generate it.")
		return pd.DataFrame()
	df = pd.read_csv(path, parse_dates=["timestamp"])  # type: ignore[arg-type]
	return df


@st.cache_resource(show_spinner=False)
def load_model(path: str):
	if not Path(path).exists():
		return None
	return joblib.load(path)


# Friendly names for equipment types
TYPE_MAP = {
	"Chiller": "Big Cooler (Chiller)",
	"Lighting": "Lights",
	"HVAC": "Air & Heat (HVAC)",
	"Elevator": "Lift",
	"Pump": "Water Pusher (Pump)",
	"boiler": "Water Heater (Boiler)",
}


def make_overview(df: pd.DataFrame) -> None:
	st.subheader("📊 Campus at a glance")
	if df.empty:
		st.info("No data yet.")
		return

	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Things", int(df.get("asset_id", pd.Series(dtype=int)).nunique()))
	col2.metric("Buildings", int(df.get("building", pd.Series(dtype=str)).nunique()))
	avg_temp = df.get("temperature_c", pd.Series([np.nan])).mean()
	avg_power = df.get("power_kw", pd.Series([np.nan])).mean()
	col3.metric("Temp °C (avg)", f"{avg_temp:.1f}" if pd.notna(avg_temp) else "–")
	col4.metric("Power kW (avg)", f"{avg_power:.1f}" if pd.notna(avg_power) else "–")

	if {"timestamp", "power_kw", "building"}.issubset(df.columns):
		st.markdown("**Energy over time**")
		fig = px.line(df.sort_values("timestamp"), x="timestamp", y="power_kw", color="building")
		fig.update_layout(xaxis_title="Time", yaxis_title="Energy (kW)")
		st.plotly_chart(fig, use_container_width=True)

	if {"vibration_g", "asset_type"}.issubset(df.columns):
		st.markdown("**How much things shake**")
		fig = px.histogram(df.assign(asset_type=df["asset_type"].map(lambda t: TYPE_MAP.get(str(t), str(t)))), x="vibration_g", nbins=40, color="asset_type")
		fig.update_layout(xaxis_title="Shake", yaxis_title="How many")
		st.plotly_chart(fig, use_container_width=True)

	st.markdown("<small>Tip: "
			"Things = equipment items. Energy shows how much power buildings use over time." 
			"</small>", unsafe_allow_html=True)


def predict_latest(df: pd.DataFrame, model) -> pd.DataFrame:
	if model is None or df.empty:
		return pd.DataFrame()
	if "timestamp" not in df.columns or "asset_id" not in df.columns:
		return pd.DataFrame()
	features = [
		"temperature_c",
		"humidity_pct",
		"vibration_g",
		"power_kw",
		"usage_hours",
		"age_years",
	]
	for col in features:
		if col not in df.columns:
			return pd.DataFrame()
	idx = df.groupby("asset_id")["timestamp"].idxmax()
	latest = df.loc[idx].copy()
	proba = model.predict_proba(latest[features].values)[:, 1]
	latest["chance_to_break"] = proba
	latest["asset_type_friendly"] = latest["asset_type"].map(lambda t: TYPE_MAP.get(str(t), str(t)))
	friendly = latest[["timestamp", "asset_id", "asset_type_friendly", "building", "chance_to_break"]].rename(
		columns={
			"timestamp": "time",
			"asset_id": "thing",
			"asset_type_friendly": "type",
			"building": "building",
			"chance_to_break": "chance_to_break",
		}
	)
	return friendly


def make_health(pred_df: pd.DataFrame) -> None:
	st.subheader("🛠️ Health & Risk")
	if pred_df.empty:
		st.info("Model or data missing.")
		return
	st.markdown("**Which things might break soon**")
	sorted_df = pred_df.sort_values("chance_to_break", ascending=False)
	st.dataframe(sorted_df, use_container_width=True)
	fig = px.bar(sorted_df.head(20), x="thing", y="chance_to_break", color="building")
	fig.update_layout(yaxis_title="Chance to break")
	st.plotly_chart(fig, use_container_width=True)

	st.markdown("<small>Tip: "
			"Chance to break = how likely a thing might have a problem soon." 
			"</small>", unsafe_allow_html=True)


def make_stream(df: pd.DataFrame, limit: int = 30, sleep: float = 0.05) -> None:
	st.subheader("🔴 Live Feed (sample)")
	if df.empty or "timestamp" not in df.columns:
		st.info("Live feed needs timestamped data.")
		return
	placeholder = st.empty()
	df_sorted = df.sort_values("timestamp").tail(limit)
	for _, row in df_sorted.iterrows():
		# Map type inline for readability
		row_dict = row.to_dict()
		row_dict["asset_type"] = TYPE_MAP.get(str(row_dict.get("asset_type", "")), str(row_dict.get("asset_type", "")))
		placeholder.json(row_dict)
		time.sleep(sleep)

	st.markdown("<small>Tip: "
			"Live feed shows the latest readings as they arrive." 
			"</small>", unsafe_allow_html=True)


def make_tasks(pred_df: pd.DataFrame) -> None:
	st.subheader("📅 Make a Task")
	if pred_df.empty:
		st.info("No suggestions yet.")
		return
	candidates = pred_df.sort_values("chance_to_break", ascending=False)["thing"].tolist()
	thing = st.selectbox("Pick a thing", candidates)
	day = st.date_input("Pick a day")
	priority = st.segmented_control("Priority", options=["LOW", "MEDIUM", "HIGH"], selection_mode="single")
	if st.button("✅ Create Task", type="primary"):
		st.success(f"Task created for {thing} on {day} ({priority})")

	st.markdown("<small>Tip: "
			"Use tasks to plan fixes for things with high chance to break." 
			"</small>", unsafe_allow_html=True)


def make_agent() -> None:
	st.subheader("🤖 AI Helper")
	sensitivity = st.slider("Warn sooner vs later", 0.1, 0.9, 0.5, 0.05)
	use_gemini = st.toggle("Use smart suggestions (Gemini)", value=True)
	if st.button("🚀 Run AI", type="primary"):
		with st.spinner("Thinking..."):
			state = run_workflow(input_csv=CSV_PATH, model_path=MODEL_PATH, risk_threshold=sensitivity, use_gemini=use_gemini)
			st.success("Done!")
			if getattr(state, "anomalies", None) is not None and not state.anomalies.empty:
				st.markdown("**Weird readings**")
				st.dataframe(state.anomalies.head(100), use_container_width=True)
			if getattr(state, "predictions", None) is not None and not state.predictions.empty:
				st.markdown("**Predictions**")
				st.dataframe(state.predictions, use_container_width=True)
			if getattr(state, "tasks", None):
				st.markdown("**Planned tasks**")
				st.json(state.tasks)
			if getattr(state, "recommendations", None):
				st.markdown("**Energy tips**")
				for r in state.recommendations:
					st.write(f"- {r}")

	st.markdown("<small>Tip: "
			"AI helper can flag weird readings and suggest energy tips." 
			"</small>", unsafe_allow_html=True)


# Sidebar quick actions
with st.sidebar:
	st.header("⚙️ Quick Actions")
	if st.button("🔄 Reload data"):
		st.cache_data.clear()
		st.rerun()


# Data + model
_df = load_csv(CSV_PATH)
_model = load_model(MODEL_PATH)
_pred = predict_latest(_df, _model)

# Tabs for simple navigation
overview_tab, health_tab, tasks_tab, live_tab, ai_tab = st.tabs([
	"🏠 Home",
	"🛠️ Health",
	"📅 Tasks",
	"🔴 Live",
	"🤖 AI",
])

with overview_tab:
	make_overview(_df)

with health_tab:
	make_health(_pred)

with tasks_tab:
	make_tasks(_pred)

with live_tab:
	with st.expander("Show feed"):
		make_stream(_df)

with ai_tab:
	make_agent()
