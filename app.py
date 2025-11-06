# app.py — HPLC-based Thalassemia Screening (NiceGUI, bilingual, CSV batch, charts)
import os, sys, subprocess, io, joblib
import pandas as pd
import numpy as np
from nicegui import ui

os.environ["WEB_CONCURRENCY"] = "1"

PORT = int(os.environ.get('PORT', 8080))
MODEL_PATH = 'artifacts/model.pkl'
DATA_PATH = 'data/HPLC data.csv'
LABELS = {0: 'Normal', 1: 'Carrier', 2: 'Disease'}  # fixed id → label

# ------------- i18n (AZ/EN) -------------------------------------------------
LANG = {'value': 'az'}  # default language

TXT = {
    'az': {
        'title': 'HPLC əsaslı Talassemiya Skrininqi',
        'brand': 'Demo • Klinik istifadə üçün deyil',
        'enter': 'Biomarkerləri daxil et',
        'preset_normal': 'Preset: Normal',
        'preset_carrier': 'Preset: Daşıyıcı',
        'preset_disease': 'Preset: Xəstə',
        'predict': 'PROQNOZ',
        'prediction': 'Proqnoz',
        'confidence': 'Etibarlılıq',
        'probs': 'Sinif ehtimalları',
        'why': 'Niyə belə?',
        'range_warn': 'Diapazondan kənar dəyərlər var:',
        'disclaimer': 'Məsuliyyət qeydi / Məhdudiyyətlər',
        'disc_text': (
            "- Bu alət **tədqiqat və tədris** məqsədlidir; klinik qərar üçün uyğun deyil.\n"
            "- Dəyərlər laboratoriya referenslərinə görə dəyişə bilər; şübhə olduqda **həkim qərarı** əsasdır.\n"
            "- “Niyə belə?” bölməsi sadə qaydalarla qeyri-formal izah verir; modelin daxili mexanizmini əvəz etmir."
        ),
        'upload_title': 'CSV yüklə (batch proqnoz)',
        'upload_hint': 'CSV faylında sütun adları sahə adları ilə uyğun olmalıdır.',
        'download': 'Nəticələri yüklə (CSV)',
        'charts_toggle': 'Qrafikləri göstər',
        'model_info': 'Model məlumatları',
        'model_name': 'Model',
        'sk_version': 'scikit-learn versiyası',
        'calibrated': 'Kalibrasiya',
        'yes': 'Bəli',
        'no': 'Xeyr',
        'cannot_start': 'Tətbiq açıla bilmədi',
        'fix': 'Həll: repoya artifacts/model.pkl yükləyin və ya data/HPLC data.csv əlavə edin ki, server bir dəfə öyrədə bilsin.',
    },
    'en': {
        'title': 'HPLC-based Thalassemia Screening',
        'brand': 'Demo • Not for clinical use',
        'enter': 'Enter patient biomarkers',
        'preset_normal': 'Preset: Normal',
        'preset_carrier': 'Preset: Carrier',
        'preset_disease': 'Preset: Disease',
        'predict': 'PREDICT',
        'prediction': 'Prediction',
        'confidence': 'Confidence',
        'probs': 'Class probabilities',
        'why': 'Why?',
        'range_warn': 'Out-of-range values:',
        'disclaimer': 'Disclaimer / Limitations',
        'disc_text': (
            "- This tool is for **research/education** only; not for clinical decisions.\n"
            "- Ranges vary by laboratory; when in doubt, rely on **physician judgement**.\n"
            "- The “Why?” block gives a rule-of-thumb explanation; it doesn’t replace the model internals."
        ),
        'upload_title': 'Upload CSV (batch inference)',
        'upload_hint': 'CSV must use the same field names as the inputs.',
        'download': 'Download results (CSV)',
        'charts_toggle': 'Show charts',
        'model_info': 'Model information',
        'model_name': 'Model',
        'sk_version': 'scikit-learn version',
        'calibrated': 'Calibrated',
        'yes': 'Yes',
        'no': 'No',
        'cannot_start': '⚠️ Cannot start the app',
        'fix': 'Fix: commit artifacts/model.pkl or include data/HPLC data.csv so the server can train once.',
    }
}

def t(key): return TXT[LANG['value']][key]

# ------------- Model yükla / lazım olsa train et ----------------------------
def load_or_train_model():
    # 1) Var olan modeli yüklə
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            return bundle['model'], bundle.get('meta', {}), None
        except Exception:
            # zədəlidirsə sil
            try: os.remove(MODEL_PATH)
            except Exception: pass
    # 2) Dataset varsa bir dəfə serverdə öyrət
    if os.path.exists(DATA_PATH):
        try:
            os.makedirs('artifacts', exist_ok=True)
            subprocess.check_call([sys.executable, 'train.py'])
            bundle = joblib.load(MODEL_PATH)
            return bundle['model'], bundle.get('meta', {}), None
        except Exception as e:
            return None, None, f"Training failed: {e}"
    # 3) Heç nə yoxdursa, istifadəçiyə mesaj
    return None, None, "Model not found and dataset missing."

model, meta, err = load_or_train_model()

# ------------- Sahələr, aralıqlar, ipucları ---------------------------------
FIELDS = {
    'HbA0':  {'label_en':'HbA0 (%)','label_az':'HbA0 (%)', 'min':0,   'max':100, 'step':0.1, 'hint_en':'Main hemoglobin fraction (HPLC).','hint_az':'Əsas hemoglobin fraksiyası (HPLC).'},
    'HbA2':  {'label_en':'HbA2 (%)','label_az':'HbA2 (%)', 'min':0,   'max':10,  'step':0.1, 'hint_en':'Typically 1.5–3.5%. High → β-thal trait.','hint_az':'Adətən 1.5–3.5%. Yüksək → β-talassemiya göstəricisi.'},
    'HbF':   {'label_en':'HbF (%)','label_az':'HbF (%)',   'min':0,   'max':40,  'step':0.1, 'hint_en':'Fetal hemoglobin. Elevated in thalassemia.','hint_az':'Fetal Hb. Talassemiyada yüksələ bilər.'},
    'RBC':   {'label_en':'RBC (10^12/L)','label_az':'RBC (10^12/L)', 'min':1,   'max':8,   'step':0.01,'hint_en':'Red cell count; often high/normal in thalassemia.','hint_az':'Eritrosit sayı; talassemiyada çox vaxt yüksək/normal.'},
    'HB':    {'label_en':'Hb (g/dL)','label_az':'Hb (g/dL)', 'min':4, 'max':20,  'step':0.1, 'hint_en':'Hemoglobin level.','hint_az':'Hemoglobin səviyyəsi.'},
    'MCV':   {'label_en':'MCV (fL)','label_az':'MCV (fL)',  'min':60, 'max':110, 'step':0.5, 'hint_en':'<80 fL microcytosis.','hint_az':'<80 fL mikrositoz.'},
    'MCH':   {'label_en':'MCH (pg)','label_az':'MCH (pg)',  'min':15, 'max':40,  'step':0.1, 'hint_en':'<27 pg hypochromia.','hint_az':'<27 pg hipoxromiya.'},
    'MCHC':  {'label_en':'MCHC (g/dL)','label_az':'MCHC (g/dL)', 'min':25,'max':38,'step':0.1,'hint_en':'Mean cell Hb concentration.','hint_az':'Orta hüceyrə Hb konsentrasiyası.'},
    'RDWcv': {'label_en':'RDW-CV (%)','label_az':'RDW-CV (%)', 'min':10,'max':25,'step':0.1,'hint_en':'Red cell size variation.','hint_az':'Eritrosit ölçü dəyişkənliyi.'},
    'S_Window': {'label_en':'S-Window (HPLC)','label_az':'S-Window (HPLC)', 'min':0,'max':5,'step':0.1,'hint_en':'Side window signal (optional).','hint_az':'Yan pəncərə siqnalı (opsional).'},
    'Unknown':  {'label_en':'Unknown (HPLC)','label_az':'Unknown (HPLC)',  'min':0,'max':5,'step':0.1,'hint_en':'Undetermined fraction (optional).','hint_az':'Təyin olunmayan fraksiya (opsional).'},
    'Age':      {'label_en':'Age (years)','label_az':'Yaş (il)',           'min':0,'max':100,'step':1, 'hint_en':'Optional.','hint_az':'Opsional.'},
}
CATS = {
    'Gender': ('Gender', ['M','F'], 'Biological sex / Bioloji cins'),
    'Weekness': ('Weakness', ['Yes','No'], 'Subjective weakness / Zəiflik şikayəti'),
    'Jaundice': ('Jaundice', ['Yes','No'], 'Jaundice / Sarılıq'),
    'Religion': ('Religion (optional)', None, 'Optional'),
    'Present_District': ('Present_District (optional)', None, 'Optional'),
}

def label_for(k):
    return FIELDS[k]['label_en'] if LANG['value']=='en' else FIELDS[k]['label_az']

def hint_for(k):
    return FIELDS[k]['hint_en'] if LANG['value']=='en' else FIELDS[k]['hint_az']

# ------------- Sadə qayda-əsaslı izah ---------------------------------------
def rule_based_explanation(row):
    msgs = []
    v = lambda x: row.get(x, None)
    try:
        if v('MCV') is not None and v('MCV') < 80: msgs.append('MCV < 80 fL → microcytosis / mikrositoz')
        if v('MCH') is not None and v('MCH') < 27: msgs.append('MCH < 27 pg → hypochromia / hipoxromiya')
        if v('HbA2') is not None and v('HbA2') > 3.5: msgs.append('HbA2 > 3.5% → thal trait sign / talassemi göstəricisi')
        if v('HbF') is not None and v('HbF') > 2.0: msgs.append('HbF > 2% → fetal Hb elevated / fetal Hb yüksək')
        if v('RBC') is not None and v('RBC') > 5.5: msgs.append('RBC high → consistent with thal / talassemi ilə uyğun')
        if not msgs: msgs.append('Profile looks borderline/normal / Profil sərhəddə və ya normaldır')
    except Exception:
        pass
    return ' • '.join(msgs)

def out_of_range_msgs(row):
    msgs = []
    for k, meta in FIELDS.items():
        v = row.get(k, None)
        if v is None or (isinstance(v, float) and np.isnan(v)): 
            continue
        if v < meta['min'] or v > meta['max']:
            lab = label_for(k)
            msgs.append(f"{lab}: {v}  ({meta['min']}–{meta['max']})")
    return msgs

# ------------- UI: başlıq, dil seçimi ---------------------------------------
with ui.header().classes('items-center justify-between bg-blue-600 text-white'):
    ui.label(t('title')).classes('text-2xl font-bold')
    with ui.row().classes('items-center'):
        ui.label('AZ / EN')
        def set_lang(v):
            LANG['value'] = v
            ui.notify(f"Language: {v.upper()}")
        ui.toggle({'az':'AZ', 'en':'EN'}, value='az', on_change=lambda e: set_lang(e.value)).props('color=white')

# ------------- Model status / info ------------------------------------------
def detect_model_meta(model, meta):
    name = meta.get('model_name')
    calibrated = meta.get('calibrated', False)
    try:
        import sklearn
        skver = sklearn.__version__
    except Exception:
        skver = 'unknown'
    # Heç bir meta yoxdursa, pipeline class adını alaq
    if name is None:
        try: name = type(model).__name__
        except Exception: name = 'unknown'
    # Kalibrasiya olub-olmadığını pipeline-dan sez
    if not calibrated:
        try:
            from sklearn.calibration import CalibratedClassifierCV
            if isinstance(getattr(model, 'named_steps', {}).get('clf', None), CalibratedClassifierCV):
                calibrated = True
        except Exception:
            pass
    return name, skver, calibrated

if err:
    with ui.card().classes('max-w-3xl mx-auto mt-10'):
        ui.label(t('cannot_start')).classes('text-xl font-semibold text-red-600')
        ui.label(err).classes('text-red-600')
        ui.label(t('fix')).classes('text-gray-700')
else:
    model_name, skver, calibrated = detect_model_meta(model, meta)

    # ---- Model info panel
    with ui.card().classes('max-w-5xl mx-auto mt-6'):
        ui.label(t('model_info')).classes('text-lg font-medium')
        with ui.grid(columns=3).classes('gap-4'):
            ui.label(f"{t('model_name')}: {model_name}")
            ui.label(f"{t('sk_version')}: {skver}")
            ui.label(f"{t('calibrated')}: {t('yes') if calibrated else t('no')}")

    # ---- Input card
    with ui.card().classes('max-w-5xl mx-auto mt-4'):
        ui.label(t('enter')).classes('text-lg font-medium mb-2')

        inputs = {}
        with ui.grid(columns=3).classes('gap-4'):
            # numeric
            for key, metaF in FIELDS.items():
                with ui.column():
                    n = ui.number(
                        label=f"{label_for(key)} ( {metaF['min']}–{metaF['max']} )",
                        min=metaF['min'], max=metaF['max'], step=metaF['step'], value=None
                    ).props('outlined dense clearable')
                    ui.icon('info').classes('text-gray-500').tooltip(hint_for(key))
                    inputs[key] = n
            # categoricals
            for k, (lab, options, tip) in CATS.items():
                if options:
                    w = ui.select(options, label=lab, value=None, with_input=True).props('outlined dense use-input fill-input')
                else:
                    w = ui.input(label=lab).props('outlined dense')
                ui.icon('info').classes('text-gray-500').tooltip(tip)
                inputs[k] = w

        # Presets
        def set_preset(kind):
            presets = {
                'Normal':  {'HbA0':96,'HbA2':2.5,'HbF':0.8,'RBC':4.8,'HB':14,'MCV':88,'MCH':29,'MCHC':34,'RDWcv':12.5,'S_Window':0,'Unknown':0,'Age':30,'Gender':'M','Weekness':'No','Jaundice':'No'},
                'Carrier': {'HbA0':96,'HbA2':3.7,'HbF':1.8,'RBC':5.5,'HB':13,'MCV':74,'MCH':24,'MCHC':33,'RDWcv':14,'S_Window':0,'Unknown':0,'Age':28,'Gender':'F','Weekness':'No','Jaundice':'No'},
                'Disease': {'HbA0':80,'HbA2':2.0,'HbF':8.0,'RBC':5.2,'HB':9.5,'MCV':68,'MCH':22,'MCHC':31,'RDWcv':18,'S_Window':1,'Unknown':0.5,'Age':10,'Gender':'M','Weekness':'Yes','Jaundice':'Yes'},
            }
            for k,v in presets[kind].items():
                c = inputs.get(k)
                if c is None: continue
                try: c.value = v
                except Exception: pass

        with ui.row().classes('gap-2 mt-2'):
            ui.button(t('preset_normal'),  on_click=lambda: set_preset('Normal')).props('flat color=primary')
            ui.button(t('preset_carrier'), on_click=lambda: set_preset('Carrier')).props('flat color=primary')
            ui.button(t('preset_disease'), on_click=lambda: set_preset('Disease')).props('flat color=primary')

        # Toggle charts
        show_charts = ui.checkbox(t('charts_toggle'), value=True)

        # Results
        res_title = ui.label().classes('text-xl font-semibold mt-4')
        res_sub   = ui.label().classes('text-sm text-gray-500')
        # bar chart for probabilities
        chart = ui.echart({
            'xAxis': {'type':'category', 'data':['Normal','Carrier','Disease']},
            'yAxis': {'type':'value', 'min':0, 'max':1},
            'series':[{'type':'bar','data':[0,0,0]}],
            'grid': {'left': 40, 'right': 10, 'bottom': 30, 'top': 10}
        }).classes('w-full h-48')
        def update_chart(ps):
            chart.options['series'][0]['data'] = ps
            chart.update()
            chart.visible = show_charts.value

        reason_box = ui.label().classes('text-sm mt-2')

        def predict():
            row = {}
            for k,c in inputs.items():
                v = getattr(c, 'value', None)
                row[k] = (None if v=='' else v)
            df = pd.DataFrame([row])

            # out-of-range warnings
            msgs = out_of_range_msgs(row)
            if msgs:
                ui.notification(t('range_warn') + '\n' + '\n'.join(msgs), type='warning', close_button=True)

            # inference
            yhat = model.predict(df)[0]
            res_title.set_text(f"{t('prediction')}: {LABELS[int(yhat)]}")

            # probabilities (calibrated olub-olmamasını info-da göstəririk)
            proba = getattr(model, 'predict_proba', None)
            if proba:
                p = proba(df)[0].tolist()
                res_sub.set_text(f"{t('confidence')}: {max(p):.2%}")
                update_chart(p)
            else:
                res_sub.set_text('')

            reason_box.set_text(f"{t('why')}: " + rule_based_explanation(row))

        ui.button(t('predict'), on_click=predict).props('unelevated color=primary').classes('mt-3')

    # ---- CSV upload (batch inference)
    with ui.card().classes('max-w-5xl mx-auto mt-4'):
        ui.label(t('upload_title')).classes('text-lg font-medium')
        ui.label(t('upload_hint')).classes('text-gray-600')
        out_table = ui.element('div')
        dl_link = ui.link(t('download'), '#').props('disable').classes('mt-2')

        def on_upload(e):
            try:
                content: bytes = e.content.read()
                df = pd.read_csv(io.BytesIO(content))
                # predict
                preds = model.predict(df)
                res = pd.DataFrame({'prediction':[LABELS[int(x)] for x in preds]})
                proba = getattr(model, 'predict_proba', None)
                if proba:
                    P = pd.DataFrame(proba(df), columns=['p_normal','p_carrier','p_disease'])
                    res = pd.concat([res, P], axis=1)
                out = pd.concat([df.reset_index(drop=True), res], axis=1)

                # show small html table preview
                html = out.head(20).to_html(index=False)
                out_table.clear()
                with out_table:
                    ui.html(f'<div class="overflow-x-auto">{html}</div>')

                # prepare download
                csv_bytes = out.to_csv(index=False).encode('utf-8')
                b64 = csv_bytes.decode('utf-8')
                # simple data URL (for small files); for large use ui.download with temp file
                from urllib.parse import quote
                dl_link.text = t('download')
                dl_link.href = 'data:text/csv;charset=utf-8,' + quote(out.to_csv(index=False))
                dl_link.props(remove='disable')

                # optional: class distribution notify
                counts = res['prediction'].value_counts().to_dict()
                ui.notification(f"Done. Distribution: {counts}", type='positive')
            except Exception as ex:
                ui.notification(str(ex), type='negative', close_button=True)

        ui.upload(on_upload=on_upload, auto_upload=True).props('accept=.csv')

    # ---- Disclaimer
    with ui.expansion(t('disclaimer')).classes('max-w-5xl mx-auto mt-4'):
        ui.markdown(t('disc_text'))

# ---- Run
ui.run(host='0.0.0.0', port=PORT, reload=False, show=False)

