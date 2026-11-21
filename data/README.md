# Data

## Included

### `raw/oxford/parkinsons.data` - Oxford Parkinson's dataset (D1)
- Source: UCI Machine Learning Repository, dataset "Parkinsons".
- Citation (as given on the UCI page): Little, M. (2007). Parkinsons [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C59C74
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Related paper: M. A. Little et al., "Suitability of dysphonia measurements for telemonitoring of Parkinson's disease," IEEE Trans. Biomed. Eng., 2009.

### `raw/istanbul/pd_speech_features.csv` - Istanbul Parkinson's dataset (D3)
- Source: UCI Machine Learning Repository, dataset "Parkinson's Disease Classification".
- Citation (as given on the UCI page): Sakar, C., Serbes, G., Gunduz, A., Nizam, H., & Sakar, B. (2018). Parkinson's Disease Classification [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MS4X
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

### `processed/italian_vowel_features.csv`, `processed/italian_reading_features.csv`
- The 11 voice features (jitter, shimmer, HNR, NHR) of the Italian recordings, measured by us with Praat through Parselmouth (`src/italian_features.py`).
- One row per recording, with `person_id`, `group` (PD / HEC = elderly healthy / YHC = young healthy) and file name.
- All training, explanation and stability steps run directly from these files.

## Not included

### `raw/italian/` - Italian Parkinson's Voice and Speech (D4)
- The .wav recordings are **not** included.
- Download them from IEEE DataPort: "Italian Parkinson's Voice and Speech" (G. Dimauro, F. Girardi), doi: 10.21227/aw6b-tg17
- Licence: Creative Commons Attribution.
- Papers the dataset author asks users to cite:
  - G. Dimauro, D. Caivano, V. Bevilacqua, F. Girardi and V. Napoletano, "VoxTester, software for digital evaluation of speech changes in Parkinson disease," 2016 IEEE International Symposium on Medical Measurements and Applications (MeMeA), 2016, pp. 1-6, doi: 10.1109/MeMeA.2016.7533761.
  - G. Dimauro, V. Di Nicola, V. Bevilacqua, D. Caivano and F. Girardi, "Assessment of Speech Intelligibility in Parkinson's Disease Using a Speech-To-Text System," IEEE Access, vol. 5, pp. 22199-22208, 2017, doi: 10.1109/ACCESS.2017.2762475.
- Unzip them into `data/raw/italian/`, so that it contains the folders `15 Young Healthy Control`, `22 Elderly Healthy Control` and `28 People with Parkinson's disease`.
- They are needed **only** to re-run feature extraction (`src/italian_features.py`), the recording check (`src/check_recordings.py`) and `app/test_app.py`.
