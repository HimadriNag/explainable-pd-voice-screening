# Check if the Italian groups were recorded in different conditions
# (sample rate, channels, duration, background noise, loudness).
import os
import numpy as np
import pandas as pd
import parselmouth

from italian_features import list_files

FRAME_SEC = 0.02  # 20 ms frames


def recording_info(wav_path):
    snd = parselmouth.Sound(wav_path)
    info = {"sample_rate": snd.sampling_frequency,
            "channels": snd.n_channels,
            "duration_s": snd.duration}

    x = snd.convert_to_mono().values[0]  # samples between -1 and 1
    n = int(FRAME_SEC * snd.sampling_frequency)
    frames = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    frame_db = 20 * np.log10(rms + 1e-10)  # dB relative to full scale (dBFS)

    info["noise_db"] = np.percentile(frame_db, 10)  # quietest 10% of frames
    info["loudness_db"] = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10)
    info["loud_minus_noise_db"] = info["loudness_db"] - info["noise_db"]
    return info


if __name__ == "__main__":
    files = list_files()
    files = files[files["kind"].isin(["vowel", "reading"])]

    rows = []
    for _, r in files.iterrows():
        info = recording_info(r["path"])
        info.update({"file": r["file"], "person_id": r["person_id"],
                     "group": r["group"], "kind": r["kind"]})
        rows.append(info)
    check = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    check.to_csv("results/recording_check.csv", index=False)
    print("Checked", len(check), "files, saved to results/recording_check.csv")

    cols = ["channels", "duration_s", "noise_db", "loudness_db", "loud_minus_noise_db"]
    for kind in ["vowel", "reading"]:
        sub = check[check["kind"] == kind]
        print("\n=== " + kind + ": median per group")
        print(sub.groupby("group")[cols].median().round(2).to_string())
        print("sample rates found:")
        for g, s in sub.groupby("group")["sample_rate"]:
            print("   ", g, sorted(s.unique().astype(int).tolist()))

    # per-person median features, then median over people (not files)
    for kind in ["vowel", "reading"]:
        feats = pd.read_csv("data/processed/italian_" + kind + "_features.csv")
        per_person = feats.groupby(["group", "person_id"])[["jitter_local", "shimmer_local", "HNR"]].median()
        print("\n=== " + kind + ": median over people")
        print(per_person.groupby("group").median().round(4).to_string())
