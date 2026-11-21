# Dengue Early-Warning System (EDU AI Engineering Hackathon 2026 – Preliminary)

Predicts ward-level dengue risk (Low / Medium / High / Critical) **2–4 weeks ahead** from weather and
multi-week rainfall lags, for 70 wards across Dhaka North/South, Chattogram, Gazipur and Narayanganj.

## Run locally (≈ 1 minute)
```bash
git clone <YOUR_REPO_URL> && cd dengue-ews
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                    # opens http://localhost:8501
```
Pre-trained models are committed (`models/`), so the app starts immediately. To retrain (~25 s):
```bash
python -m src.train
```

## Repo map
| Path | Purpose |
|---|---|
| `src/features.py` | Per-ward rainfall lags (1–8 wk), rolling sums, temp/humidity windows, targets |
| `src/train.py` | Chronological split, LightGBM point + 10/90% quantile models, metrics |
| `app.py` | Streamlit dashboard (ward forecast, drivers, ward ranking + heatmap, validation) |
| `data/dengue.csv` | Provided dataset (weekly, ward-level, 2016–2025) |
| `ARCHITECTURE.md` | Data flow, split, model choice, cost controls |

## Results (test = 2024–2025, never used for training)
See `models/metrics.csv` or the dashboard "Model validation" tab.

## AI & external-API disclosure
- AI assistants used: **Claude (Anthropic)** – <FILL IN what each member used it for>; <add ChatGPT/Copilot/etc. if used>
- External APIs / paid services at runtime: **none** (offline inference, no LLM calls, no weather API calls).
- Libraries: pandas, numpy, scikit-learn, LightGBM, joblib, Streamlit, Plotly.

## Team
<Names + roles>
