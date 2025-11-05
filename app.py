# app.py
import os, joblib, pandas as pd
from nicegui import ui

MODEL_PATH = 'artifacts/model.pkl'
LABELS = {0: 'Normal', 1: 'Carrier', 2: 'Disease'}

bundle = joblib.load(MODEL_PATH)
pipe = bundle['model']

# Sadə forma: əsas göstəricilər (istəsən hamsını əlavə edə bilərsən)
numeric_fields = [
    'HbA0','HbA2','HbF','RBC','HB','MCV','MCH','MCHC','RDWcv'
]
optional_numeric = ['S_Window','Unknown','Age']
categorical_fields = ['Gender','Religion','Present_District','Weekness','Jaundice']

with ui.header().classes('items-center justify-between'):
    ui.label('HPLC-based Thalassemia Screening').classes('text-2xl font-bold')
    ui.label('NiceGUI • Heroku')

with ui.card().classes('max-w-3xl mx-auto mt-6'):
    ui.label('Enter patient biomarkers').classes('text-lg font-medium mb-2')

    inputs = {}
    with ui.grid(columns=3).classes('gap-4'):
        for f in numeric_fields:
            inputs[f] = ui.number(label=f, value=0.0, format='%.2f')
        for f in optional_numeric:
            inputs[f] = ui.number(label=f, value=None, format='%.2f')

        # categoricals (sadə dəyərlər; lazım gələrsə genişləndir)
        inputs['Gender'] = ui.select(['M','F'], label='Gender', value=None, with_input=True)
        inputs['Religion'] = ui.input(label='Religion (optional)')
        inputs['Present_District'] = ui.input(label='Present_District (optional)')
        inputs['Weekness'] = ui.select(['Yes','No'], label='Weekness', value=None, with_input=True)
        inputs['Jaundice'] = ui.select(['Yes','No'], label='Jaundice', value=None, with_input=True)

    result = ui.label().classes('text-xl font-semibold mt-2')
    sub = ui.label().classes('text-sm text-gray-500')

    def predict():
        row = {k: (v.value if hasattr(v, 'value') else None) for k,v in inputs.items()}
        df = pd.DataFrame([row])
        yhat = pipe.predict(df)[0]
        proba = getattr(pipe, 'predict_proba', lambda x: None)(df)
        result.set_text(f'Prediction: {LABELS[int(yhat)]}')
        if proba is not None:
            p = proba[0][int(yhat)]
            sub.set_text(f'Confidence: {p:.2%}')
        else:
            sub.set_text('')

        # sadə qrafik: əhəmiyyətli sahələri göstərmək əvəzinə dəyərləri listelə
        with details:
            details.clear()
            with ui.column():
                ui.label('Entered values').classes('text-md font-medium')
                for k in numeric_fields:
                    ui.label(f'{k}: {row.get(k)}')

    ui.button('Predict', on_click=predict).classes('mt-2')

with ui.expansion('Details').classes('max-w-3xl mx-auto mt-2') as details:
    ui.label('No details yet.')

# Heroku portu
PORT = int(os.environ.get('PORT', 8080))
ui.run(host='0.0.0.0', port=PORT)
