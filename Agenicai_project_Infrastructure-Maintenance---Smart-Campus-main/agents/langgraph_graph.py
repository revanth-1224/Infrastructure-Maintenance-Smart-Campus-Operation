from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import os
import pandas as pd
from langgraph.graph import END, StateGraph

try:
	from dotenv import load_dotenv  # type: ignore
	load_dotenv()
except Exception:
	pass

try:
	import google.generativeai as genai  # optional
except Exception:  # pragma: no cover
	genai = None  # type: ignore


@dataclass
class AgentState:
	"""Shared state passed between graph nodes."""
	data: Optional[pd.DataFrame] = None
	anomalies: Optional[pd.DataFrame] = None
	predictions: Optional[pd.DataFrame] = None
	tasks: List[Dict[str, Any]] = field(default_factory=list)
	recommendations: List[str] = field(default_factory=list)
	model_path: str = "models/model.joblib"
	input_csv: str = "data/iot_readings.csv"
	risk_threshold: float = 0.5
	use_gemini: bool = True


def ingest_node(state: AgentState) -> AgentState:
	"""Load recent sensor data from CSV if not already provided in state."""
	if state.data is None:
		try:
			state.data = pd.read_csv(state.input_csv, parse_dates=["timestamp"])  # latest
		except FileNotFoundError:
			raise RuntimeError(f"Sensor CSV not found at {state.input_csv}. Generate data first.")
	return state


def anomaly_node(state: AgentState) -> AgentState:
	"""Detect anomalies using rolling z-score and hard thresholds."""
	df = state.data.copy()
	if df is None or df.empty:
		state.anomalies = pd.DataFrame()
		return state

	metrics = ["temperature_c", "vibration_g", "power_kw"]
	anom_rows = []
	for asset_id, grp in df.sort_values("timestamp").groupby("asset_id"):
		for metric in metrics:
			roll = grp[metric].rolling(24, min_periods=6)
			mean = roll.mean()
			sd = roll.std().fillna(1.0)
			z = (grp[metric] - mean) / (sd + 1e-6)

			# Hard threshold condition per metric (Series)
			if metric == "temperature_c":
				hard = grp[metric] > 30
			elif metric == "vibration_g":
				hard = grp[metric] > 1.0
			elif metric == "power_kw":
				hard = grp[metric] > 20
			else:
				hard = pd.Series(False, index=grp.index)

			flags = (z.abs() > 3) | hard
			for idx in flags[flags].index:
				row = grp.loc[idx]
				anom_rows.append({
					"timestamp": row["timestamp"],
					"asset_id": asset_id,
					"metric": metric,
					"value": row[metric],
					"z_score": float(z.loc[idx]),
				})

	state.anomalies = pd.DataFrame(anom_rows)
	return state


def predict_node(state: AgentState) -> AgentState:
	"""Load model and compute failure probabilities per asset on latest readings."""
	df = state.data
	if df is None or df.empty:
		state.predictions = pd.DataFrame()
		return state

	try:
		model = joblib.load(state.model_path)
	except FileNotFoundError:
		raise RuntimeError("Model not found. Train it with ml/train_predictive_model.py first.")

	features = [
		"temperature_c",
		"humidity_pct",
		"vibration_g",
		"power_kw",
		"usage_hours",
		"age_years",
	]

	idx = df.groupby("asset_id")["timestamp"].idxmax()
	latest = df.loc[idx].copy()
	proba = model.predict_proba(latest[features].values)[:, 1]
	latest["failure_probability"] = proba
	state.predictions = latest[["timestamp", "asset_id", "asset_type", "building", "failure_probability"]]
	return state


def scheduler_node(state: AgentState) -> AgentState:
	"""Create maintenance tasks for high-risk assets."""
	if state.predictions is None or state.predictions.empty:
		return state

	threshold = state.risk_threshold
	for _, row in state.predictions.iterrows():
		if row["failure_probability"] >= threshold:
			state.tasks.append({
				"asset_id": row["asset_id"],
				"building": row["building"],
				"priority": "HIGH",
				"reason": f"Predicted failure risk {row['failure_probability']:.2f}",
				"scheduled_for": str(pd.Timestamp(row["timestamp"]) + pd.Timedelta(days=1)),
			})
	return state


def _gemini_recommendations(prompt: str) -> List[str]:
	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key or genai is None:
		return []
	try:
		genai.configure(api_key=api_key)
		model = genai.GenerativeModel("gemini-1.5-flash")
		resp = model.generate_content(prompt)
		text = resp.text or ""
		lines = [ln.strip("- ") for ln in text.splitlines() if ln.strip()]
		return [ln for ln in lines if len(ln) > 0][:5]
	except Exception:
		return []


def optimizer_node(state: AgentState) -> AgentState:
	"""Suggest energy efficiency actions. If configured, call Gemini for richer suggestions."""
	recs: List[str] = []
	if state.anomalies is not None and not state.anomalies.empty:
		vib_anoms = state.anomalies[state.anomalies["metric"] == "vibration_g"]["asset_id"].unique()
		for aid in vib_anoms:
			recs.append(f"Inspect bearings and alignment for {aid} due to high vibration.")

	mean_power = pd.Series(dtype=float)
	if state.data is not None and not state.data.empty:
		mean_power = state.data.groupby("building")["power_kw"].mean().sort_values(ascending=False)
		for building, val in mean_power.head(2).items():
			recs.append(f"Evaluate load scheduling in {building} (avg {val:.1f} kW).")

	if state.use_gemini:
		context = {
			"top_buildings": mean_power.head(3).to_dict() if not mean_power.empty else {},
			"num_anomalies": int(len(state.anomalies)) if state.anomalies is not None else 0,
		}
		prompt = (
			"You are an energy optimization assistant for a university campus. "
			"Given context and typical HVAC/lighting best practices, give 3-5 concise, actionable recommendations. "
			f"Context: {context}. Avoid generic statements; be specific."
		)
		llm_recs = _gemini_recommendations(prompt)
		recs.extend(llm_recs)

	state.recommendations = recs
	return state


def build_graph():
	graph = StateGraph(AgentState)
	graph.add_node("ingest_node", ingest_node)
	graph.add_node("anomaly_node", anomaly_node)
	graph.add_node("predict_node", predict_node)
	graph.add_node("scheduler_node", scheduler_node)
	graph.add_node("optimizer_node", optimizer_node)

	graph.set_entry_point("ingest_node")
	graph.add_edge("ingest_node", "anomaly_node")
	graph.add_edge("anomaly_node", "predict_node")
	graph.add_edge("predict_node", "scheduler_node")
	graph.add_edge("scheduler_node", "optimizer_node")
	graph.add_edge("optimizer_node", END)
	return graph.compile()


def run_workflow(
	input_csv: str = "data/iot_readings.csv",
	model_path: str = "models/model.joblib",
	risk_threshold: float = 0.5,
	data: Optional[pd.DataFrame] = None,
	use_gemini: bool = True,
) -> AgentState:
	"""Execute the LangGraph agent and return the final state as AgentState."""
	app = build_graph()
	init_state = AgentState(
		input_csv=input_csv,
		model_path=model_path,
		risk_threshold=risk_threshold,
		data=data,
		use_gemini=use_gemini,
	)
	result = app.invoke(init_state.__dict__)
	if isinstance(result, dict):
		try:
			return AgentState(**result)
		except TypeError:
			# Fallback: attach known keys only
			return AgentState(
				data=result.get("data"),
				anomalies=result.get("anomalies"),
				predictions=result.get("predictions"),
				tasks=result.get("tasks", []),
				recommendations=result.get("recommendations", []),
				model_path=result.get("model_path", model_path),
				input_csv=result.get("input_csv", input_csv),
				risk_threshold=result.get("risk_threshold", risk_threshold),
				use_gemini=result.get("use_gemini", use_gemini),
			)
	return result  # type: ignore[return-value]
