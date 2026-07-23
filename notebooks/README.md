# Showcase notebooks

These demos use only `docs/showcase/` (a few MB of metrics + scores).

```powershell
python -m pip install -e ".[notebooks]"
jupyter notebook
```

1. `01_project_walkthrough.ipynb` — story + splits  
2. `02_detection_results.ipynb` — PR/ROC efficacy  
3. `03_clinical_reality_check.ipynb` — clinical + portfolio self-eval  

If figures are missing, from the repo root:

```powershell
python scripts/make_showcase_assets.py
```
