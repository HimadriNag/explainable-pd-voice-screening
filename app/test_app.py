# Quick check of the app without the browser: one PD and one elderly vowel file.
# NOTE: both files were in the training data, so this only checks that the app runs.
import sys
import pandas as pd

sys.path.insert(0, "app")
from app import predict, risk_level

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)

TESTS = [("PD (Domenico C)",
          "data/raw/italian/28 People with Parkinson's disease/1-5/Domenico C/VA1cdaopmoe67M2605161906.wav"),
         ("Elderly (Agnese P)",
          "data/raw/italian/22 Elderly Healthy Control/AGNESE P/VA1APGANRET55F170320171107.wav")]

for name, path in TESTS:
    prob, table = predict(path)
    print("==", name, " score = %.3f (%s)" % (prob, risk_level(prob)))
    print(table.head(3).to_string(index=False))
    print()
