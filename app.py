# app.py (defensive)

import os
os.environ.setdefault("WEB_CONCURRENCY", "1")

import os, sys, subprocess, joblib
import pandas as pd
from nicegui import ui

PORT = int(os.environ.get('PORT', 8080))
MODEL_PATH = 'artifacts/model.pkl'
LABELS = {0: 'Normal', 1: 'Carrier', 2: 'Disease'}

def try_load_model():
    try:
        if os.path.exists(MODEL_PATH):
            bundle = joblib.load(MODEL_PATH)
            return bundle['model'], None
        else:
            if os.path.exists('data/HPLC data.csv'):
                try:
                    os.makedirs('artifacts', exist_ok=True)
                    subprocess.check_call([sys.executable, 'train.py'])
                    bundle = joblib.load(MODEL_PATH)
                    return bundle['model'], None
                except Exception as e:
                    return None, f"Training failed on server: {e}"
            return None, "Model not found and no dataset to train on."
    except Exception as e:
        return None, f"Model load error: {e}"

model, err = try_load_model()

with ui.header().classes('items-center justify-between'):
    ui.label('HPLC-based Thalassemia Screening').classes('text-2xl font-bold')
    ui.label('NiceGUI • Heroku')

if err:
    with ui.card().classes('max-w-3xl mx-auto mt-8'):
        ui.label('⚠️ Cannot start the app').classes('text-xl font-semibold')
        ui.label(err).classes('text-red-600')
        ui.label('Fix: commit artifacts/model.pkl OR include data/HPLC data.csv in repo').classes('text-gray-600')
else:
    pipe = model
    fields_num = ['HbA0','HbA2','HbF','RBC','HB','MCV','MCH','MCHC','RDWcv']
    fields_num_opt = ['S_Window','Unknown','Age']
    with ui.card().classes('max-w-3xl mx-auto mt-6'):
        ui.label('Enter patient biomarkers').classes('text-lg font-medium mb-2')
        inputs = {}
        with ui.grid(columns=3).classes('gap-4'):
            for f in fields_num:
                inputs[f] = ui.number(label=f, value=0.0, format='%.2f').props('outlined dense')
            for f in fields_num_opt:
                inputs[f] = ui.number(label=f, value=None, format='%.2f').props('outlined dense')
            inputs['Gender'] = ui.select(['M','F'], label='Gender', value=None, with_input=True).props('outlined dense use-input')
            inputs['Religion'] = ui.input(label='Religion (optional)').props('outlined dense')
            inputs['Present_District'] = ui.input(label='Present_District (optional)').props('outlined dense')
            inputs['Weekness'] = ui.select(['Yes','No'], label='Weekness', value=None, with_input=True).props('outlined dense use-input')
            inputs['Jaundice'] = ui.select(['Yes','No'], label='Jaundice', value=None, with_input=True).props('outlined dense use-input')

        out = ui.label().classes('text-xl font-semibold mt-2')
        sub = ui.label().classes('text-sm text-gray-500')

        def predict():
            row = {k: (v.value if hasattr(v, 'value') else None) for k,v in inputs.items()}
            df = pd.DataFrame([row])
            try:
                yhat = pipe.predict(df)[0]
                proba = getattr(pipe, 'predict_proba', None)
                out.set_text(f'Prediction: {LABELS[int(yhat)]}')
                if proba:
                    p = proba(df)[0][int(yhat)]
                    sub.set_text(f'Confidence: {p:.2%}')
                else:
                    sub.set_text('')
            except Exception as e:
                out.set_text('Prediction error')
                sub.set_text(str(e))

        ui.button('Predict', on_click=predict).props('unelevated color=primary').classes('mt-2')

ui.run(host='0.0.0.0', port=PORT, reload=False, show=False)

