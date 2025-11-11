Agentic Smart Campus Manager
A full-stack educational demo that simulates a smart campus with fake IoT sensors, trains a predictive maintenance model, provides a modern Streamlit dashboard, and orchestrates an AI workflow using LangGraph.

Features
Synthetic IoT sensor data generator (CSV + optional streaming sample)
Predictive maintenance (RandomForest) trained on synthetic data
Streamlit dashboard with a simple, kid-friendly UI: tabs, big buttons, emojis, and clear wording
Dark theme with accessible contrast (custom .streamlit/config.toml)
LangGraph agentic workflow with ingest, anomaly detection, prediction, scheduling, and energy optimization nodes
Optional Gemini-powered recommendations in the optimizer node when GEMINI_API_KEY is set (supports .env)
Project Structure
.
├─ data/
│  └─ generate_fake_iot.py
├─ ml/
│  └─ train_predictive_model.py
├─ agents/
│  └─ langgraph_graph.py
├─ app/
│  ├─ streamlit_app.py
│  └─ startup.py             # Auto-generate CSV and train model if missing
├─ models/
│  └─ model.joblib (generated)
├─ tests/
│  ├─ test_data.py
│  └─ test_model.py
├─ .streamlit/
│  └─ config.toml           # Theme (dark, indigo accent)
├─ .gitignore               # Ignores OS/editor, env, Node/Python artifacts
├─ requirements.txt
├─ Dockerfile               # Launches startup.py
├─ render.yaml
└─ README.md
Tech Stack
UI: Streamlit + Plotly
ML: scikit-learn (RandomForest), pandas, numpy, joblib
Agent: LangGraph (with optional Google Gemini)
Utilities: python-dotenv for .env, pyarrow for Parquet support (optional)
UI Overview (Simple Tabs)
The dashboard is organized into 5 tabs for clarity:

🏠 Home: Quick metrics and charts (energy over time, shaking histogram)
🛠️ Health: “Chance to break” for each thing (equipment)
📅 Tasks: Create maintenance tasks for high-risk things
🔴 Live: Sample live feed of incoming readings
🤖 AI: Run the agent to flag weird readings and suggest energy tips
Friendly Words Used in the UI
“thing”: equipment item (e.g., a light or pump)
“type”: what kind of thing it is
“chance to break”: how likely a thing may have a problem soon
“weird reading”: a value that looks unusual and may need attention
Equipment Type Mapping (for easier understanding)
Chiller → Big Cooler (Chiller)
Lighting → Lights
HVAC → Air & Heat (HVAC)
Elevator → Lift
Pump → Water Pusher (Pump)
boiler → Water Heater (Boiler)
Quickstart
1) Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/activate
# bash/zsh
# source .venv/bin/activate
2) Install dependencies
pip install -r requirements.txt
3) Generate data and train model
python data/generate_fake_iot.py --output data/iot_readings.csv --assets 25 --hours 72
python ml/train_predictive_model.py --input data/iot_readings.csv --model models/model.joblib
4) (Optional) Configure Gemini
Create a Google AI Studio key.
Create a file named .env in the project root with:
GEMINI_API_KEY=YOUR_KEY_HERE
Alternatively, export the variable in your shell:
# PowerShell
$env:GEMINI_API_KEY = "YOUR_KEY_HERE"
# bash/zsh
export GEMINI_API_KEY="YOUR_KEY_HERE"
The agent’s optimizer node will use Gemini (model gemini-1.5-flash) to enrich energy recommendations.
5) Run the Streamlit app
streamlit run app/streamlit_app.py
Deployment
Render (recommended)
Uses the Dockerfile, which starts app/startup.py. On first boot, it creates data/iot_readings.csv and trains models/model.joblib if they’re missing, then serves Streamlit on port 8080.
Steps:

Push this repo to GitHub.
Render → New → Web Service → connect repo → Runtime: Docker → Create Web Service.
Add env vars if needed (e.g., GEMINI_API_KEY).
Open the live URL.
Redeploy: push to main (auto-deploy if enabled) or click Manual Deploy.

Persistent data options:

Free plan rebuilds the container on deploys; generated CSV/model are recreated on boot.
For persistence across deploys, store data in an external object store (e.g., S3) or a database, and load on startup.
Vercel (via Docker)
Add vercel.json with a Docker build, then vercel --prod. See earlier section for details.
Using the App
Home: See total things, buildings, average temperature and power. Explore energy and shaking charts.
Health: Sort by “chance to break” to find things that may need fixing soon.
Tasks: Pick a thing, pick a day, choose priority, and create a task.
Live: Watch a small sample of the latest readings.
AI: Click “Run AI” to detect weird readings, view predictions, see planned tasks, and get energy tips.
LangGraph Agent
The agent orchestrates 5 nodes:

ingest_node: Reads latest sensor data (CSV or provided dataframe)
anomaly_node: Flags unusual values
predict_node: Estimates chance to break for each thing using the trained model
scheduler_node: Suggests tasks for high-risk things
optimizer_node: Suggests energy efficiency actions; can use Gemini for richer suggestions
Run from within the Streamlit app via the “Run AI” button, or programmatically:

python -c "from agents.langgraph_graph import run_workflow; run_workflow()"
Notes
This is an educational demo; not production-ready.
Replace the synthetic data and simplistic heuristics with real integrations and domain logic as needed.
The UI theme and wording are designed to be approachable; feel free to adjust in app/streamlit_app.py and .streamlit/config.toml.
