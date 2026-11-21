# Train ONE final Logistic Regression on ALL the D4 data (Italian vowel, PD vs elderly) for the app.
# No test split here: the honest scores are already in results/table2_performance.csv.
import os
import joblib

from harness import get_task, make_pipe
from models import make_model
from data_loading import FEATURES

OUT = "app/model.joblib"

X, y, groups = get_task("D4")
pipe = make_pipe(make_model("LogReg", seed=0))  # same settings as in train_models.py
pipe.fit(X, y)

print("trained on", len(X), "recordings from", len(set(groups)), "people")
print("PD recordings:", int(y.sum()), " elderly recordings:", int((y == 0).sum()))

# save the training data too: SHAP needs it as the "background" (average voice)
os.makedirs("app", exist_ok=True)
joblib.dump({"pipe": pipe, "X_train": X, "y_train": y, "features": FEATURES}, OUT)
print("saved to", OUT)
