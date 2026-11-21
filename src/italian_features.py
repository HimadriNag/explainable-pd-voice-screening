# Measure the 11 voice features from the Italian .wav files with Praat (parselmouth).
import os
import re
import glob
import pandas as pd
import parselmouth
from parselmouth.praat import call

from data_loading import FEATURES

ITALIAN_DIR = "data/raw/italian"
OUT_DIR = "data/processed"

# Same name, different person: this Vito S is 47 (file code VSIOTLOP47M);
# the other Vito S folders (6-10, 11-16) are one man aged 46 recorded twice.
RENAME_PERSON = {os.path.join("17-28", "Vito S"): "Vito S (47)"}


def voice_features(wav_path):
    # some PD files are 44100 Hz, the rest 16000 Hz -> make them all 16000 Hz mono
    snd = parselmouth.Sound(wav_path).convert_to_mono().resample(16000)
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)

    f = {}
    f["jitter_local"] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    f["jitter_abs"] = call(pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    f["jitter_rap"] = call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    f["jitter_ppq5"] = call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

    f["shimmer_local"] = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    f["shimmer_db"] = call([snd, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    f["shimmer_apq3"] = call([snd, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    f["shimmer_apq5"] = call([snd, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    f["shimmer_apq11"] = call([snd, pp], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    harm = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    f["HNR"] = call(harm, "Get mean", 0, 0)

    # NHR is only given in Praat's Voice report, so read it from the report text
    pitch = call(snd, "To Pitch", 0.0, 75, 500)
    report = call([snd, pitch, pp], "Voice report", 0, 0, 75, 500, 1.3, 1.6, 0.03, 0.45)
    m = re.search(r"Mean noise-to-harmonics ratio:\s*([-\d.eE]+)", report)
    f["NHR"] = float(m.group(1)) if m else float("nan")

    return f


def group_from_folder(folder):
    name = folder.lower()
    if "young" in name:
        return "YHC"
    if "elderly" in name:
        return "HEC"
    if "parkinson" in name:
        return "PD"
    return None


def list_files():
    # one row per .wav file: path, group, person_id, kind (vowel / reading / None)
    rows = []
    for group_folder in sorted(os.listdir(ITALIAN_DIR)):
        group = group_from_folder(group_folder)
        if group is None:
            continue
        group_path = os.path.join(ITALIAN_DIR, group_folder)
        # PD people sit one level deeper (inside 1-5, 6-10, ...), so search recursively
        for wav in sorted(glob.glob(os.path.join(group_path, "**", "*.wav"), recursive=True)):
            person_dir = os.path.relpath(os.path.dirname(wav), group_path)
            person = RENAME_PERSON.get(person_dir, os.path.basename(person_dir))
            fname = os.path.basename(wav).upper()
            if fname.startswith(("VA1", "VA2")):
                kind = "vowel"
            elif fname.startswith(("B1", "B2")):
                kind = "reading"
            else:
                kind = None
            rows.append({"path": wav, "file": os.path.basename(wav), "group": group,
                         "person_id": "D4_" + group + "_" + person, "kind": kind})
    return pd.DataFrame(rows)


def extract(files):
    rows, failed = [], []
    for _, r in files.iterrows():
        try:
            f = voice_features(r["path"])
            bad = [k for k in FEATURES if pd.isna(f[k])]
            if bad:
                raise ValueError("undefined: " + ", ".join(bad))
        except Exception as e:
            failed.append((r["file"], str(e)))
            continue
        f.update({"person_id": r["person_id"], "group": r["group"], "file": r["file"]})
        rows.append(f)
    return pd.DataFrame(rows, columns=FEATURES + ["person_id", "group", "file"]), failed


if __name__ == "__main__":
    files = list_files()

    # 1) summary first, so we can check the files before extracting
    print("SUMMARY (before extraction)")
    for kind in ["vowel", "reading"]:
        sub = files[files["kind"] == kind]
        summ = sub.groupby("group").agg(people=("person_id", "nunique"), files=("file", "count"))
        print("\n" + kind + ":")
        print(summ.to_string())

    # 2) extract and save
    os.makedirs(OUT_DIR, exist_ok=True)
    for kind in ["vowel", "reading"]:
        sub = files[files["kind"] == kind]
        table, failed = extract(sub)
        out = os.path.join(OUT_DIR, "italian_" + kind + "_features.csv")
        table.to_csv(out, index=False)
        print("\n=== " + kind + ": saved " + str(len(table)) + " files to " + out)
        print("failed: " + str(len(failed)))
        for name, err in failed:
            print("   ", name, "->", err)

        # 3) average of three key features per group
        print(table.groupby("group")[["jitter_local", "shimmer_local", "HNR"]].mean().round(4).to_string())
