# Anumaan — Complete Decision-Support Prototype

## What is included
- Real ML training pipeline on explicitly synthetic operational history.
- Rolling 7-day demand forecast with learned day-of-week patterns.
- Day-by-day projected inventory, stockout window and recommended replenishment quantity.
- Live public weather from Open-Meteo (no API key for the public endpoint used here).
- Public OpenStreetMap/Overpass road context.
- Public OSRM candidate routing with an offline-safe fallback.
- Separate ML route-risk classifier trained on synthetic disruption scenarios.
- Risk-aware route decision (risk weighted more than ETA).
- Demand and severe-weather what-if demonstrations.
- Dispatch priority based on forecasted stockout urgency.

## Data honesty
Representative public cities are used only for demonstration. Operational demand/inventory history and route-disruption labels are synthetic/anonymized. The application does not claim access to confidential military data, and route-risk values are prototype outputs rather than operational estimates.

## Run on Windows PowerShell
From the extracted `Anumaan_Complete` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python backend\train.py
python -m uvicorn backend.main:app --reload --port 8000
```

Open a second PowerShell window:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. API docs are at `http://127.0.0.1:8000/docs`.

## Best demo sequence
1. **Forecast** — choose a start date that moves through Fri/Sat/Sun, run forecast, and show predicted demand + projected inventory.
2. Point out **stockout window** and **recommended dispatch quantity**.
3. **Route Intelligence** — choose two different representative regions and calculate alternatives. Toggle severe-weather what-if and recalculate.
4. **Dispatch** — show forecast-driven urgency queue.
5. **What-If** — increase demand and show how additional records become critical.
6. **Model** — show validation metrics and explain that synthetic labels are kept separate from live/public inputs.

## Architecture
Synthetic operational history -> Demand ML -> 7-day inventory projection -> stockout window -> replenishment quantity -> public candidate routes -> synthetic-scenario route-risk ML -> risk-aware route decision.
