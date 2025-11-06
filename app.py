# app.py — HPLC-based Thalassemia Screening (NiceGUI, AZ/EN, range dropdowns, charts)
import os, sys, subprocess, io, joblib
import pandas as pd
import numpy as np
from nicegui import ui

# Heroku: uvicorn "workers" xətasının qarşısı üçün
os.environ["WEB_CONCURRENCY"] = "1"

PORT = int(os.environ.get('PORT', 8080))
MODEL_PATH = 'artifacts/model.pkl'
DATA_PATH = 'data/HPLC data.csv'
LABELS = {0: 'Normal', 1: 'Carrier', 2: 'Disease'}

# ---------------- i18n (AZ/EN) ----------------
LANG = {'value': 'az'}  # default

TXT = {
    'az': {
        'title': 'HPLC əsaslı Talassemiya Skrininqi',
        'brand': 'Demo • Klinik istifadə üçün deyil',
        'enter': 'Biomarkerləri daxil et',
        'ranges': 'Aralıqlar',
        'manual': 'Manual dəyər',
        'preset_normal': 'Preset: Normal',
        'preset_carrier': 'Preset: Daşıyıcı',
        'preset_disease': 'Preset: Xəstə',
        'clear': 'Təmizlə',
        'predict': 'PROQNOZ',
        'prediction': 'Proqnoz',
        'confidence': 'Etibarlılıq',
        'why': 'Niyə belə?',
        'charts_toggle': 'Qrafikləri göstər',
        'range_warn': 'Diapazondan kənar dəyərlər:',
        'model_info': 'Model məlumatları',
        'model_name': 'Model',
        'sk_version': 'scikit-learn versiyası',
        'calibrated': 'Kalibrasiya',
        'yes': 'Bəli', 'no': 'Xeyr',
        'cannot_start': 'Tətbiq açıla bilmədi',
        'fix': 'Həll: repoya artifacts/model.pkl yükləyin və ya data/HPLC data.csv əlavə edin ki, server bir dəfə öyrədə bilsin.',
        'disclaimer': 'Məsuliyyət qeydi / Məhdudiyyətlər',
        'disc_text': (
            "- Bu alət **tədqiqat və tədris** məqsədlidir; klinik qərar üçün uyğun deyil.\n"
            "- Dəyərlər laborator laboratoriyadan asılı dəyişə bilər; şübhədə qalanda **həkim qərarı** əsasdır.\n"
            "- “Niyə belə?” bölməsi sadə qaydalarla izah verir; modelin daxili mexanizmini əvəz etmir."
        ),
    },
    'en': {
        'title': 'HPLC-based Thalassemia Screening',
        'brand': 'Demo • Not for clinical use',
        'enter': 'Enter patient biomarkers',
        'ranges': 'Ranges',
        'manual': 'Manual value',
        'preset_normal': 'Preset: Normal',
        'preset_carrier': 'Preset: Carrier',
        'preset_disease': 'Preset: Disease',
        'clear': 'Clear',
        'predict': 'PREDICT',
        'prediction': 'Prediction',
        'confidence': 'Confidence',
        'why': 'Why?',
        'charts_toggle': 'Show charts',
        'range_warn': 'Out-of-range values:',
        'model_info': 'Model information',
        'model_name': 'Model',
        'sk_version': 'scikit-learn version',
        'calibrated': 'Calibrated',
        'yes': 'Yes', 'no': 'No',
        'cannot_start': '⚠️ Cannot start the app',
        'fix': 'Fix: commit artifacts/model.pkl or include data/HPLC data.csv so the server can train once.',
        'disclaimer': 'Disclaimer / Limitations',
        'disc_text': (
            "- This tool is for **research/education** only; not for clinical decisions.\n"
            "- Reference ranges vary by lab; when in doubt rely on **physician judgement**.\n"
            "- The “Why?” block is a rule-of-thumb explanation; it doesn’t replace model internals."
        ),
    }
}
def t(k): return TXT[LANG['value']][k]

# --------------- model load/train ----------------
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            b = joblib.load(MODEL_PATH)
            return b['model'], b.get('meta', {}), None
        except Exception:
            try: os.remove(MODEL_PATH)
            except: pass
    if os.path.exists(DATA_PATH):
        try:
            os.makedirs('artifacts', exist_ok=True)
            subprocess.check_call([sys.executable, 'train.py'])
            b = joblib.load(MODEL_PATH)
            return b['model'], b.get('meta', {}), None
        except Exception as e:
            return None, None, f"Training failed: {e}"
    return None, None, "Model not found and dataset missing."
model, meta, err = load_or_train_model()

# ------------- fields & clinical bins -------------
FIELDS = {
    'HbA0':  {'label': {'az':'HbA0 (%)','en':'HbA0 (%)'},  'min':0, 'max':100, 'step':0.1,
              'bins': [(0,90),(90,97),(97,100)], 'hints': {'az':'Əsas Hb fraksiyası','en':'Main Hb fraction'}},
    'HbA2':  {'label': {'az':'HbA2 (%)','en':'HbA2 (%)'},  'min':0, 'max':10,  'step':0.1,
              'bins': [(0,2.0),(2.0,3.5),(3.5,10)], 'hints': {'az':'1.5–3.5 normal, >3.5 daşıyıcı göstəricisi','en':'1.5–3.5 normal, >3.5 carrier sign'}},
    'HbF':   {'label': {'az':'HbF (%)','en':'HbF (%)'},    'min':0, 'max':40,  'step':0.1,
              'bins': [(0,2.0),(2.0,10),(10,40)], 'hints': {'az':'>2% yüksəlmiş ola bilər','en':'>2% can be elevated'}},
    'RBC':   {'label': {'az':'RBC (10^12/L)','en':'RBC (10^12/L)'}, 'min':1, 'max':8, 'step':0.01,
              'bins': [(1,4.5),(4.5,5.5),(5.5,8)], 'hints': {'az':'Talassemiyada çox vaxt yüksək/normal','en':'Often high/normal in thal'}},
    'HB':    {'label': {'az':'Hb (g/dL)','en':'Hb (g/dL)'}, 'min':4, 'max':20, 'step':0.1,
              'bins': [(4,10),(10,12),(12,20)], 'hints': {'az':'Hemoglobin səviyyəsi','en':'Hemoglobin level'}},
    'MCV':   {'label': {'az':'MCV (fL)','en':'MCV (fL)'},  'min':60,'max':110,'step':0.5,
              'bins': [(60,75),(75,80),(80,110)], 'hints': {'az':'<80 fL mikrositoz','en':'<80 fL microcytosis'}},
    'MCH':   {'label': {'az':'MCH (pg)','en':'MCH (pg)'},  'min':15,'max':40,'step':0.1,
              'bins': [(15,24),(24,27),(27,40)], 'hints': {'az':'<27 pg hipoxromiya','en':'<27 pg hypochromia'}},
    'MCHC':  {'label': {'az':'MCHC (g/dL)','en':'MCHC (g/dL)'}, 'min':25,'max':38,'step':0.1,
              'bins': [(25,31),(31,34),(34,38)], 'hints': {'az':'Orta hüceyrə Hb konsentr.','en':'Mean cell Hb conc.'}},
    'RDWcv': {'label': {'az':'RDW-CV (%)','en':'RDW-CV (%)'}, 'min':10,'max':25,'step':0.1,
              'bins': [(10,13),(13,16),(16,25)], 'hints': {'az':'Eritrosit ölçü dəyişkənliyi','en':'Red cell size variation'}},
    'S_Window': {'label': {'az':'S-Window (HPLC)','en':'S-Window (HPLC)'}, 'min':0,'max':5,'step':0.1,
                 'bins': [(0,0.5),(0.5,1.5),(1.5,5)], 'hints': {'az':'Opsional siqnal','en':'Optional signal'}},
    'Unknown':  {'label': {'az':'Unknown (HPLC)','en':'Unknown (HPLC)'}, 'min':0,'max':5,'step':0.1,
                 'bins': [(0,0.5),(0.5,1.5),(1.5,5)], 'hints': {'az':'Opsional','en':'Optional'}},
    'Age':      {'label': {'az':'Yaş (il)','en':'Age (years)'}, 'min':0,'max':100,'step':1,
                 'bins': [(0,12),(12,40),(40,100)], 'hints': {'az':'Opsional','en':'Optional'}},
}
CATS = {
    'Gender': ('Gender', ['M','F']),
    'Weekness': ('Weakness', ['Yes','No']),
    'Jaundice': ('Jaundice', ['Yes','No']),
    'Religion': ('Religion (optional)', None),
    'Present_District': ('Present_District (optional)', None),
}

# ------------- helpers -------------
def label_for(k): return FIELDS[k]['label'][LANG['value']]
def hint_for(k):  return FIELDS[k]['hints'][LANG['value']]

def rule_based_explanation(row):
    msgs = []
    v = lambda x: row.get(x, None)
    try:
        if v('MCV') is not None and v('MCV') < 80: msgs.append('MCV < 80 fL → microcytosis / mikrositoz')
        if v('MCH') is not None and v('MCH') < 27: msgs.append('MCH < 27 pg → hypochromia / hipoxromiya')
        if v('HbA2') is not None and v('HbA2') > 3.5: msgs.append('HbA2 > 3.5% → carrier sign / daşıyıcı göstəricisi')
        if v('HbF') is not None and v('HbF') > 2.0: msgs.append('HbF > 2% → fetal Hb yüksək')
        if v('RBC') is not None and v('RBC') > 5.5: msgs.append('RBC yüksək → talassemiya ilə uyğun')
        if not msgs: msgs.append('Profil sərhəddə/normal görünür')
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
            msgs.append(f"{label_for(k)}: {v}  ({meta['min']}–{meta['max']})")
    return msgs

# ------------- header (language) -------------
with ui.header().classes('items-center justify-between bg-blue-600 text-white'):
    ui.label(lambda: t('title')).classes('text-2xl font-bold')
    with ui.row().classes('items-center'):
        ui.label('AZ / EN')
        def set_lang(v):
            LANG['value'] = v
            # sadə yol: tüm mətni yeniləmək üçün reload
            ui.run_javascript('location.reload()')
        ui.toggle({'az':'AZ','en':'EN'}, value='az', on_change=lambda e: set_lang(e.value)).props('color=white')

# ------------- status / model info -------------
def detect_model_meta(model, meta):
    name = meta.get('model_name')
    calibrated = meta.get('calibrated', False)
    try:
        import sklearn; skver = sklearn.__version__
    except Exception:
        skver = 'unknown'
    if name is None:
        try: name = type(model).__name__
        except: name = 'unknown'
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
        ui.label(lambda: t('cannot_start')).classes('text-xl font-semibold text-red-600')
        ui.label(err).classes('text-red-600')
        ui.label(lambda: t('fix')).classes('text-gray-700')
else:
    model_name, skver, calibrated = detect_model_meta(model, meta)

    with ui.card().classes('max-w-5xl mx-auto mt-6'):
        ui.label(lambda: t('model_info')).classes('text-lg font-medium')
        with ui.grid(columns=3).classes('gap-4'):
            ui.label(lambda: f"{t('model_name')}: {model_name}")
            ui.label(lambda: f"{t('sk_version')}: {skver}")
            ui.label(lambda: f"{t('calibrated')}: {t('yes') if calibrated else t('no')}")

    # ------------- inputs card -------------
    with ui.card().classes('max-w-5xl mx-auto mt-4'):
        ui.label(lambda: t('enter')).classes('text-lg font-medium mb-2')

        inputs_num = {}
        inputs_bin = {}
        inputs_cat = {}

        with ui.grid(columns=3).classes('gap-4'):
            # Numeric with range dropdown + manual
            for k, metaF in FIELDS.items():
                with ui.column():
                    ui.label(label_for(k)).classes('text-sm')
                    # range dropdown (bins → "a–b")
                    options = ['—'] + [f"{a}–{b}" for a,b in metaF['bins']]
                    dd = ui.select(options, value='—').props('outlined dense')
                    # manual input
                    num = ui.number(
                        label=lambda: f"{t('manual')} ( {metaF['min']}–{metaF['max']} )",
                        min=metaF['min'], max=metaF['max'], step=metaF['step'], value=None
                    ).props('outlined dense clearable')
                    ui.icon('info').classes('text-gray-500').tooltip(hint_for(k))

                    # when dropdown changes → fill midpoint to manual box
                    def on_dd_change(e, key=k, meta=metaF, box=num):
                        val = e.value
                        if val and '–' in val:
                            a,b = val.split('–')
                            try:
                                a=float(a); b=float(b)
                                box.value = round((a+b)/2, 3)
                            except: pass
                    dd.on('update:model-value', on_dd_change)

                    inputs_bin[k] = dd
                    inputs_num[k] = num

            # Categorical
            for k,(lab, options) in CATS.items():
                if options:
                    w = ui.select(options, label=lab, value=None, with_input=True).props('outlined dense use-input fill-input')
                else:
                    w = ui.input(label=lab).props('outlined dense')
                inputs_cat[k] = w

        # presets + clear
        def set_preset(kind):
            presets = {
                'Normal':  {'HbA0':96,'HbA2':2.5,'HbF':0.8,'RBC':4.8,'HB':14,'MCV':88,'MCH':29,'MCHC':34,'RDWcv':12.5,'S_Window':0,'Unknown':0,'Age':30,'Gender':'M','Weekness':'No','Jaundice':'No'},
                'Carrier': {'HbA0':96,'HbA2':3.7,'HbF':1.8,'RBC':5.5,'HB':13,'MCV':74,'MCH':24,'MCHC':33,'RDWcv':14,'S_Window':0,'Unknown':0,'Age':28,'Gender':'F','Weekness':'No','Jaundice':'No'},
                'Disease': {'HbA0':80,'HbA2':2.0,'HbF':8.0,'RBC':5.2,'HB':9.5,'MCV':68,'MCH':22,'MCHC':31,'RDWcv':18,'S_Window':1,'Unknown':0.5,'Age':10,'Gender':'M','Weekness':'Yes','Jaundice':'Yes'},
            }
            data = presets[kind]
            for k,v in data.items():
                if k in inputs_num: inputs_num[k].value = v
                if k in inputs_cat and v in (CATS.get(k,[None,[]])[1] or []):
                    inputs_cat[k].value = v
            # dropdown-ları sıfırla
            for k in inputs_bin: inputs_bin[k].value = '—'

        def clear_all():
            for c in inputs_num.values(): c.value = None
            for c in inputs_cat.values(): c.value = None
            for c in inputs_bin.values(): c.value = '—'

        with ui.row().classes('gap-2 mt-2'):
            ui.button(lambda: t('preset_normal'),  on_click=lambda: set_preset('Normal')).props('flat color=primary')
            ui.button(lambda: t('preset_carrier'), on_click=lambda: set_preset('Carrier')).props('flat color=primary')
            ui.button(lambda: t('preset_disease'), on_click=lambda: set_preset('Disease')).props('flat color=primary')
            ui.button(lambda: t('clear'), on_click=clear_all).props('flat')

        # charts toggle
        show_charts = ui.checkbox(lambda: t('charts_toggle'), value=True)

        # results
        res_title = ui.label().classes('text-xl font-semibold mt-4')
        res_sub   = ui.label().classes('text-sm text-gray-500')
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
            # prefer manual if given, else take midpoint from dropdown (already filled)
            row = {}
            for k in FIELDS.keys():
                v = inputs_num[k].value
                row[k] = (None if v=='' else v)
            for k in inputs_cat.keys():
                row[k] = inputs_cat[k].value if inputs_cat[k].value != '' else None

            df = pd.DataFrame([row])

            # range warning
            msgs = out_of_range_msgs(row)
            if msgs:
                ui.notification(t('range_warn') + '\n' + '\n'.join(msgs), type='warning', close_button=True)

            yhat = model.predict(df)[0]
            res_title.set_text(f"{t('prediction')}: {LABELS[int(yhat)]}")

            proba = getattr(model, 'predict_proba', None)
            if proba:
                p = proba(df)[0].tolist()
                res_sub.set_text(f"{t('confidence')}: {max(p):.2%}")
                update_chart(p)
            else:
                res_sub.set_text('')
                chart.visible = False

            reason_box.set_text(f"{t('why')}: " + rule_based_explanation(row))

        ui.button(lambda: t('predict'), on_click=predict).props('unelevated color=primary').classes('mt-3')

    # disclaimer
    with ui.expansion(lambda: t('disclaimer')).classes('max-w-5xl mx-auto mt-4'):
        ui.markdown(lambda: t('disc_text'))

ui.run(host='0.0.0.0', port=PORT, reload=False, show=False)
