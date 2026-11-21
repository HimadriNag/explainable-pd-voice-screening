# Gradio app: record or upload a steady "aaaa" -> voice-risk score + SHAP explanation.
import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # draw charts without opening a window
import matplotlib.pyplot as plt
import gradio as gr

# run from the project folder; src/ holds our feature code
sys.path.insert(0, "src")
from italian_features import voice_features

DISCLAIMER = ("Research prototype for screening only. NOT a medical diagnosis. "
              "Trained on one Italian dataset with known recording differences.")

# 1) load the trained model and its training data
saved = joblib.load("app/model.joblib")
pipe = saved["pipe"]
FEATURES = saved["features"]
prep = pipe[:-1]               # imputer + scaler
model = pipe.named_steps["model"]

# SHAP works on the scaled numbers the model really sees
background = prep.transform(saved["X_train"])
explainer = shap.LinearExplainer(model, background)

# 2) research tags from our own results
core = pd.read_csv("results/table5_stable_core.csv")
STABLE = set(core.loc[core["stable"], "feature"])
direction = pd.read_csv("results/table_direction_D4.csv")
OPPOSITE = set(direction.loc[~direction["matches_textbook"], "feature"])


def risk_level(p):
    # simple cut-offs we chose ourselves (not clinically validated)
    if p < 1 / 3:
        return "low"
    if p < 2 / 3:
        return "medium"
    return "high"


def research_tag(feat):
    tags = []
    if feat in STABLE:
        tags.append("stable across runs")
    if feat in OPPOSITE:
        tags.append("warning: opposite to textbook in our data (likely recording effect)")
    return "; ".join(tags)


def predict(wav_path):
    # voice_features makes the sound mono 16000 Hz and measures the 11 features
    f = voice_features(wav_path)
    X = pd.DataFrame([f])[FEATURES]
    if X.isna().any().any():
        bad = list(X.columns[X.isna().iloc[0]])
        raise ValueError("Could not measure: " + ", ".join(bad) +
                         ". Please record a steady 'aaaa' for 3-5 seconds.")

    prob = pipe.predict_proba(X)[0, 1]
    push = explainer.shap_values(prep.transform(X))[0]  # + = towards PD, - = towards healthy

    table = pd.DataFrame({"feature": FEATURES,
                          "value": X.iloc[0].values,
                          "SHAP push": push,
                          "research tag": [research_tag(c) for c in FEATURES]})
    table = table.reindex(table["SHAP push"].abs().sort_values(ascending=False).index)
    return prob, table


def shap_chart(table):
    t = table.iloc[::-1]  # biggest push at the top
    colors = ["tab:red" if v > 0 else "tab:blue" for v in t["SHAP push"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(t["feature"], t["SHAP push"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP push (log-odds)   <- towards healthy | towards PD ->")
    ax.set_title("Why this recording got its score")
    fig.tight_layout()
    return fig


def analyse(wav_path):
    if wav_path is None:
        return "Please record or upload a .wav file first.", None, None
    try:
        prob, table = predict(wav_path)
    except Exception as e:
        return "Error: " + str(e), None, None
    score = "Voice-risk score: %.2f  (%s)\n\n%s" % (prob, risk_level(prob), DISCLAIMER)
    show = table.copy()
    show["value"] = show["value"].map(lambda v: "%.5g" % v)
    show["SHAP push"] = show["SHAP push"].round(3)
    return score, shap_chart(table), show


with gr.Blocks(title="PD voice screening prototype") as demo:
    gr.Markdown("# Voice-based Parkinson's screening (research prototype)")
    gr.Markdown("**" + DISCLAIMER + "**")
    gr.Markdown("Say a steady **'aaaa'** for 3-5 seconds, then press *Analyse*.")
    audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Your voice")
    button = gr.Button("Analyse")
    score_out = gr.Textbox(label="Result", lines=3)
    plot_out = gr.Plot(label="SHAP: which features pushed the score")
    table_out = gr.Dataframe(label="Feature details")
    button.click(analyse, inputs=audio, outputs=[score_out, plot_out, table_out])


if __name__ == "__main__":
    demo.launch()
