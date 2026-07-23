# Showcase assets

Lightweight files for the README and notebooks (no raw EEG).

| Path | Contents |
|---|---|
| `metrics/` | JSON/CSV headlines + compact `test_scores_smooth15.npz` |
| `figures/` | PNGs embedded in the root README |

Regenerate from local `outputs/` (when available):

```powershell
python scripts/make_showcase_assets.py
```
