# Load the Oxford and Istanbul datasets with the same 11 feature names.
import os
import pandas as pd

# 11 shared feature names (same order for every dataset)
FEATURES = ["jitter_local", "jitter_abs", "jitter_rap", "jitter_ppq5",
            "shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11",
            "HNR", "NHR"]

OXFORD_COLS = {
    "MDVP:Jitter(%)": "jitter_local",
    "MDVP:Jitter(Abs)": "jitter_abs",
    "MDVP:RAP": "jitter_rap",
    "MDVP:PPQ": "jitter_ppq5",
    "MDVP:Shimmer": "shimmer_local",
    "MDVP:Shimmer(dB)": "shimmer_db",
    "Shimmer:APQ3": "shimmer_apq3",
    "Shimmer:APQ5": "shimmer_apq5",
    "MDVP:APQ": "shimmer_apq11",
    "HNR": "HNR",
    "NHR": "NHR",
}

ISTANBUL_COLS = {
    "locPctJitter": "jitter_local",
    "locAbsJitter": "jitter_abs",
    "rapJitter": "jitter_rap",
    "ppq5Jitter": "jitter_ppq5",
    "locShimmer": "shimmer_local",
    "locDbShimmer": "shimmer_db",
    "apq3Shimmer": "shimmer_apq3",
    "apq5Shimmer": "shimmer_apq5",
    "apq11Shimmer": "shimmer_apq11",
    "meanHarmToNoiseHarmonicity": "HNR",
    "meanNoiseToHarmHarmonicity": "NHR",
}


def load_oxford(path="data/raw/oxford/parkinsons.data"):
    df = pd.read_csv(path)
    df = df.rename(columns=OXFORD_COLS)
    # name looks like phon_R01_S01_1 -> person code is the 3rd part (S01)
    df["person_id"] = "D1_" + df["name"].str.split("_").str[2]
    df["label"] = df["status"]
    return df[FEATURES + ["person_id", "label"]]


def load_istanbul(path="data/raw/istanbul/pd_speech_features.csv"):
    df = pd.read_csv(path)
    # some copies of this file have an extra title row on top
    if "id" not in df.columns:
        df = pd.read_csv(path, header=1)
    df = df.rename(columns=ISTANBUL_COLS)
    # keep only what we need (the file has 755 columns)
    df = df[FEATURES + ["id", "class"]].copy()
    df["person_id"] = "D3_" + df["id"].astype(str)
    df["label"] = df["class"]
    return df[FEATURES + ["person_id", "label"]]


def audit(name, df):
    # one label per person (take the first recording's label)
    person_labels = df.groupby("person_id")["label"].first()
    per_person = df.groupby("person_id").size()
    return {
        "dataset": name,
        "rows": len(df),
        "people": df["person_id"].nunique(),
        "recordings_per_person_min": per_person.min(),
        "recordings_per_person_max": per_person.max(),
        "pd_people": int((person_labels == 1).sum()),
        "healthy_people": int((person_labels == 0).sum()),
        "missing_values": int(df.isna().sum().sum()),
    }


if __name__ == "__main__":
    rows = [audit("Oxford (D1)", load_oxford()),
            audit("Istanbul (D3)", load_istanbul())]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    table.to_csv("results/table1_audit.csv", index=False)
    print("\nSaved to results/table1_audit.csv")
