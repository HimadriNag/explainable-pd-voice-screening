# An Explainable Voice-Based Screening System for Parkinson's Disease with Age-Aware Validation

Developed at the Department of Electronics and Communication Engineering, National Institute of Technology Calicut, India.

**Author:** Himadri Nag

## Summary

We study how far voice-based machine learning for Parkinson's disease (PD) screening can be trusted. On three public datasets (Oxford, Istanbul and an Italian vowel dataset) we use the same 11 voice features (jitter, shimmer, HNR, NHR), five models and ten random seeds, always splitting the data by person. We measure (1) how stable SHAP and permutation-importance explanations are across seeds, models and datasets, and (2) how much of a task can be solved by recording conditions alone, using a recording-only baseline. We find that random record-wise splits inflate accuracy, that explanations are stable across seeds but not across models and datasets, and that on the Italian data the recording setup alone separates young from elderly controls, so only PD vs elderly is a fair comparison. A small Gradio app shows a voice-risk score with its SHAP explanation and warnings from these results. It is a research prototype for screening only.

## Folder structure

```
app/            Gradio screening app (app.py), saved model (model.joblib), quick test (test_app.py)
data/           datasets - see data/README.md
  processed/    Italian features measured by us (included)
  raw/oxford/   Oxford dataset (included)
  raw/istanbul/ Istanbul dataset (included)
  raw/italian/  Italian .wav recordings (NOT included - download, see data/README.md)
results/        all result tables (.csv) and figures (.png/.pdf)
src/            all project code
tests/          leakage test and sanity test
run_app.bat     double-click to start the app (Windows)
requirements.txt
```

## Setup

```
conda create -n pdvoice python=3.11
conda activate pdvoice
pip install -r requirements.txt
```

Run every command from this folder (the one containing this README).

## Reproducing the results

Run the commands in this order:

1. `python src/data_loading.py`
2. `python src/italian_features.py` (needs the Italian .wav download)
3. `python src/check_recordings.py` (needs the Italian .wav download)
4. `python src/recording_baseline.py`
5. `python tests/test_leakage.py` and `python tests/test_sanity.py`
6. `python src/train_models.py`
7. `python src/explain.py` (about 20 min)
8. `python src/stability.py`
9. `python src/train_final.py`
10. `python app/app.py` (or double-click `run_app.bat`), then open http://127.0.0.1:7860 in Chrome
11. Optional: `python app/test_app.py`, a quick app test on two Italian files (needs the Italian .wav download)

Steps 2 and 3 are already done: their outputs (`data/processed/*.csv`, `results/recording_check.csv`) are included, so all other steps run directly from the included data.

| Result file | What it shows | Script / command | Needs Italian .wav download? |
|---|---|---|---|
| `results/table1_audit.csv` | Dataset audit: rows, people, recordings per person, missing values (Oxford, Istanbul) | `python src/data_loading.py` | no |
| `results/recording_check.csv` | Sample rate, duration, noise and loudness of every Italian recording | `python src/check_recordings.py` | yes |
| `results/table_recording_baseline.csv` | Baseline that uses only recording properties (no voice features) | `python src/recording_baseline.py` | no |
| `results/scores.csv` | Every fold result (all tasks, models, seeds, honest and leaky split) | `python src/train_models.py` | no |
| `results/table2_performance.csv` | Mean performance per task and model, honest split | `python src/train_models.py` | no |
| `results/table3_protocol_gap.csv` | Balanced accuracy, honest vs leaky split, and the gap | `python src/train_models.py` | no |
| `results/fig_protocol_gap.png` / `.pdf` | Bar chart of honest vs leaky balanced accuracy | `python src/train_models.py` | no |
| `results/importances.csv` | SHAP and permutation importance per task, seed, fold, model and feature | `python src/explain.py` | no |
| `results/rankings.csv` | Feature rankings from those importances | `python src/explain.py` | no |
| `results/table_direction_D4.csv` | D4: does a higher feature value push towards PD, and does that match the textbook | `python src/stability.py` | no |
| `results/table4_stability.csv` | Ranking agreement across seeds, models, datasets and methods, vs chance | `python src/stability.py` | no |
| `results/table5_stable_core.csv` | Features that are in the top 5 in most runs | `python src/stability.py` | no |
| `results/fig_stability.png` / `.pdf` | Plot of ranking agreement (RBO) vs chance | `python src/stability.py` | no |
| `app/model.joblib` | Final model used by the app | `python src/train_final.py` | no |

Note: microphone recording in the app works in a normal browser such as Chrome or Edge.

## Main results

All numbers are copied from the files in `results/`. Balanced accuracy, person-wise 5-fold cross-validation, mean ± SD over 10 seeds.

**Best model per dataset** (`table2_performance.csv`, `table3_protocol_gap.csv`)

| Dataset | Best model | Balanced acc. (honest) | ROC-AUC | Leaky split (same model) |
|---|---|---|---|---|
| Oxford (D1) | SVM | 0.68 ± 0.03 | 0.74 | 0.75 |
| Istanbul (D3) | LogReg | 0.67 ± 0.00 | 0.74 | 0.68 |
| Italian vowel, PD vs elderly (D4) | LogReg | 0.87 ± 0.01 | 0.93 | 0.87 |

The largest leakage gap was D1 MLP: +0.25 balanced accuracy with the leaky record-wise split.

**Recording-only baseline** (noise, loudness, duration only; `table_recording_baseline.csv`)

| Recording-only task | Balanced acc. |
|---|---|
| vowel: PD vs HEC | 0.68 ± 0.02 |
| reading: PD vs YHC | 0.98 ± 0.00 |
| reading: PD vs HEC | 0.70 ± 0.01 |
| reading: HEC vs YHC | 1.00 ± 0.00 |

(HEC = elderly healthy controls, YHC = young healthy controls.)

**Explanation stability** (rank-biased overlap with 95% bootstrap CI; chance = 0.45; `table4_stability.csv`)

| Level | SHAP RBO | Permutation RBO |
|---|---|---|
| seed | 0.89 [0.88, 0.89] | 0.80 [0.79, 0.81] |
| model | 0.65 [0.60, 0.70] | 0.69 [0.64, 0.74] |
| dataset | 0.60 [0.53, 0.68] | 0.58 [0.50, 0.65] |

SHAP vs permutation agreement: RBO 0.73 [0.71, 0.75]. Stable core (top 5 in at least 80% of runs, `table5_stable_core.csv`): only `shimmer_apq11` (91% SHAP, 91% permutation).

**Shortcut** (`table_direction_D4.csv`): in the D4 Logistic Regression model, 6 of 11 features push the opposite way to the textbook (shimmer_local, shimmer_db, shimmer_apq3, shimmer_apq5, HNR, NHR), so a cleaner voice is pushed towards PD - most likely a recording effect.

## Limitations

- Small datasets (32, 252 and 47 people), one language each.
- In the Italian data, age and recording setup are mixed, so PD-related voice changes cannot be separated from ageing.
- Oxford and Istanbul features were provided precomputed; we computed the Italian ones, so measurement differences add to dataset differences.
- Only 11 classic features; KernelExplainer SHAP values for SVM and MLP are approximate.
- The app is trained on one Italian dataset, its low / medium / high cut-offs are our own choice, and it has not been tested with real users or clinicians. It is a research prototype for screening only.

## Dataset citations

- Oxford: Little, M. (2007). Parkinsons [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59C74 (CC BY 4.0). Related paper: M. A. Little et al., "Suitability of dysphonia measurements for telemonitoring of Parkinson's disease," IEEE Trans. Biomed. Eng., 2009.
- Istanbul: Sakar, C., Serbes, G., Gunduz, A., Nizam, H., & Sakar, B. (2018). Parkinson's Disease Classification [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MS4X (CC BY 4.0).
- Italian: G. Dimauro, F. Girardi, "Italian Parkinson's Voice and Speech," IEEE DataPort, doi: 10.21227/aw6b-tg17 (Creative Commons Attribution). Papers the dataset author asks users to cite:
  - G. Dimauro, D. Caivano, V. Bevilacqua, F. Girardi and V. Napoletano, "VoxTester, software for digital evaluation of speech changes in Parkinson disease," 2016 IEEE International Symposium on Medical Measurements and Applications (MeMeA), 2016, pp. 1-6, doi: 10.1109/MeMeA.2016.7533761.
  - G. Dimauro, V. Di Nicola, V. Bevilacqua, D. Caivano and F. Girardi, "Assessment of Speech Intelligibility in Parkinson's Disease Using a Speech-To-Text System," IEEE Access, vol. 5, pp. 22199-22208, 2017, doi: 10.1109/ACCESS.2017.2762475.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The datasets are not covered by it: they keep their own licences, listed in `data/README.md`.

## Acknowledgments

Thanks to Aman Raj for help with collecting the datasets.
