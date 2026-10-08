# AdShield

Fraudulent lead and ad-click detection system.

## Setup

Requires Python 3.11+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Layout

| Path | Purpose |
| --- | --- |
| `data/` | Raw and processed datasets (gitignored except `.gitkeep`) |
| `src/adshield/data/` | Data loading and validation |
| `src/adshield/features/` | Feature engineering |
| `src/adshield/models/` | Training and inference |
| `src/adshield/api/` | FastAPI app |
| `configs/` | YAML config for model params and thresholds |
| `notebooks/` | EDA only, never production code |
| `tests/` | Test suite |

## Status

Scaffolding only; modules contain docstrings describing their eventual purpose.
