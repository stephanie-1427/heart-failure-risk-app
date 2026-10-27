import joblib
import pandas as pd
import gradio as gr

bundle = joblib.load("model.joblib")
model, features, medians = bundle["model"], bundle["features"], bundle["medians"]

def predict_risk(age, ejection_fraction, serum_creatinine, serum_sodium,
                 high_blood_pressure, anaemia, diabetes, smoking):
    row = dict(medians)  # anything not asked in the form defaults to the training median
    row.update(age=age, ejection_fraction=ejection_fraction,
               serum_creatinine=serum_creatinine, serum_sodium=serum_sodium,
               high_blood_pressure=int(high_blood_pressure), anaemia=int(anaemia),
               diabetes=int(diabetes), smoking=int(smoking))
    p = float(model.predict_proba(pd.DataFrame([row])[features])[:, 1][0])
    band = "🟢 Lower" if p < 0.3 else ("🟡 Moderate" if p < 0.6 else "🔴 Higher")
    note = (f"**{band} risk band** (demo cut-offs). Educational demo trained on 299 patients "
            "from one region. **Not medical advice.**")
    return {"Higher risk": p, "Lower risk": 1 - p}, note

demo = gr.Interface(
    fn=predict_risk,
    inputs=[
        gr.Slider(40, 95, value=60, step=1, label="Age (years)"),
        gr.Slider(10, 80, value=38, step=1, label="Ejection fraction (%)"),
        gr.Slider(0.5, 9.5, value=1.1, step=0.1, label="Serum creatinine (mg/dL)"),
        gr.Slider(113, 148, value=137, step=1, label="Serum sodium (mEq/L)"),
        gr.Checkbox(label="High blood pressure"),
        gr.Checkbox(label="Anaemia"),
        gr.Checkbox(label="Diabetes"),
        gr.Checkbox(label="Smoker"),
    ],
    outputs=[gr.Label(num_top_classes=2, label="Estimated risk"), gr.Markdown()],
    title="Heart Failure Risk Explorer (educational demo)",
    description="Enter a hypothetical patient. The model estimates risk of death during follow-up.",
)

if __name__ == "__main__":
    demo.launch()
