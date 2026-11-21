# The 5 models, with default-style settings (no tuning).
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

MODEL_NAMES = ["LogReg", "SVM", "RF", "XGB", "MLP"]


def make_model(name, seed):
    if name == "LogReg":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
    if name == "SVM":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=seed, n_jobs=1)
    if name == "XGB":
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=seed, n_jobs=1, eval_metric="logloss")
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=3000,
                             random_state=seed)
    raise ValueError("unknown model: " + name)


def get_score(pipe, X):
    # score for ROC-AUC: SVM has no probabilities here, so use its decision_function
    if hasattr(pipe, "predict_proba") and pipe.named_steps["model"].__class__.__name__ != "SVC":
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)
