# app.py — HPLC-based Thalassemia Screening (AZ only, pretty UI, multi-charts)
import os, sys, subprocess, joblib
import pandas as pd
import numpy as np
from nicegui import ui

# Heroku uvicorn "workers" problemi üçün
os.environ["WEB_CONCURRENCY"] = "1"

PORT = int(os.environ.get('PORT', 8080))
MODEL_PATH = 'artifacts/model.pkl'
DATA_PATH = 'data/HPLC data.csv'
LABELS = {0: 'Normal', 1: 'Carrier', 2: 'Disease'}

# ---------------- Model yüklə / lazım olsa serverdə bir dəfə öyrət ----------------
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

# ---------------- Sahələr, aralıqlar və qısa izahlar ----------------
FIELDS = {
    'HbA0':  {'label':'HbA0 (%)',          'min':0,  'max':100, 'step':0.1,
              'bins': [(0,90),(90,97),(97,100)], 'hint':'Əsas hemoglobin fraksiyası (HPLC).'},
    'HbA2':  {'label':'HbA2 (%)',          'min':0,  'max':10,  'step':0.1,
              'bins': [(0,2.0),(2.0,3.5),(3.5,10)], 'hint':'Adətən 1.5–3.5%. >3.5% daşıyıcılıq əlaməti ola bilər.'},
    'HbF':   {'label':'HbF (%)',           'min':0,  'max':40,  'step':0.1,
              'bins': [(0,2.0),(2.0,10),(10,40)],  'hint':'Fetal Hb. Yüksəkliyi talassemiya ilə uyğun ola bilər.'},
    'RBC':   {'label':'RBC (10^12/L)',     'min':1,  'max':8,   'step':0.01,
              'bins': [(1,4.5),(4.5,5.5),(5.5,8)], 'hint':'Eritrosit sayı. Talassemiyada bəzən yüksək/normal.'},
    'HB':    {'label':'Hb (g/dL)',         'min':4,  'max':20,  'step':0.1,
              'bins': [(4,10),(10,12),(12,20)],    'hint':'Hemoglobin səviyyəsi.'},
    'MCV':   {'label':'MCV (fL)',          'min':60, 'max':110, 'step':0.5,
              'bins': [(60,75),(75,80),(80,110)],  'hint':'<80 fL mikrositoz.'},
    'MCH':   {'label':'MCH (pg)',          'min':15, 'max':40,  'step':0.1,
              'bins': [(15,24),(24,27),(27,40)],   'hint':'<27 pg hipoxromiya.'},
    'MCHC':  {'label':'MCHC (g/dL)',       'min':25, 'max':38,  'step':0.1,
              'bins': [(25,31),(31,34),(34,38)],   'hint':'Orta hüceyrə Hb konsentrasiyası.'},
    'RDWcv': {'label':'RDW-CV (%)',        'min':10, 'max':25,  'step':0.1,
              'bins': [(10,13),(13,16),(16,25)],   'hint':'Eritrosit ölçü dəyişkənliyi.'},
    'S_Window': {'label':'S-Window (HPLC)','min':0,  'max':5,   'step':0.1,
              'bins': [(0,0.5),(0.5,1.5),(1.5,5)], 'hint':'Yan pəncərə siqnalı (opsional).'},
    'Unknown':  {'label':'Unknown (HPLC)', 'min':0,  'max':5,   'step':0.1,
              'bins': [(0,0.5),(0.5,1.5),(1.5,5)], 'hint':'Təyin olunmayan fraksiya (opsional).'},
    'Age':      {'label':'Yaş (il)',       'min':0,  'max':100, 'step':1,
              'bins': [(0,12),(12,40),(40,100)],   'hint':'Opsional.'},
}
CATS = {
    'Gender': ('Gender', ['M','F']),
    'Weekness': ('Weakness', ['Yes','No']),
    'Jaundice': ('Jaundice', ['Yes','No']),
    'Religion': ('Religion (optional)', None),
    'Present_District': ('Present_District (optional)', None),
}

# ---------------- Köməkçi funksiyalar ----------------
def rule_based_explanation(row: dict) -> str:
    msgs = []
    v = lambda x: row.get(x, None)
    try:
        if v('MCV') is not None and v('MCV') < 80: msgs.append('MCV < 80 fL → mikrositoz')
        if v('MCH') is not None and v('MCH') < 27: msgs.append('MCH < 27 pg → hipoxromiya')
        if v('HbA2') is not None and v('HbA2') > 3.5: msgs.append('HbA2 > 3.5% → daşıyıcılıq göstəricisi')
        if v('HbF') is not None and v('HbF') > 2.0: msgs.append('HbF > 2% → fetal Hb yüksək')
        if v('RBC') is not None and v('RBC') > 5.5: msgs.append('RBC yüksək → talassemiya ilə uyğun ola bilər')
        if not msgs: msgs.append('Profil sərhəddə/normal görünür')
    except Exception:
        pass
    return ' • '.join(msgs)

def out_of_range_msgs(row: dict):
    msgs = []
    for k, meta in FIELDS.items():
        v = row.get(k, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if v < meta['min'] or v > meta['max']:
            msgs.append(f"{meta['label']}: {v}  ({meta['min']}–{meta['max']})")
    return msgs

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

# ---------------- Başlıq / Hero ----------------
with ui.header().classes('items-center justify-between bg-gradient-to-r from-blue-600 to-indigo-600 text-white'):
    ui.label('HPLC əsaslı Talassemiya Skrininqi').classes('text-2xl font-bold')
    ui.label('Demo • Klinik istifadə üçün deyil')

# ---------------- Yuxarı məlumat kartı (layihə + dəyişənlər) ----------------
with ui.card().classes('max-w-6xl mx-auto mt-6 shadow-xl rounded-xl'):
    ui.label('Layihə haqqında qısa məlumat').classes('text-lg font-semibold')
    ui.markdown(
        "**Məqsəd:** HPLC və qan göstəricilərinə əsasən talassemiya statusunun (Normal / Carrier / Disease) proqnozlaşdırılması.\n\n"
        "**Qeyd:** Bu alət tədqiqat və tədris məqsədlidir; klinik qərar üçün uyğun deyil."
    ).classes('text-gray-700')
    ui.separator()
    ui.label('Göstəricilər və mənaları').classes('text-base font-medium mt-2')
    bullets = []
    for k, meta in FIELDS.items():
        bullets.append(f"- **{meta['label']}**: {meta['hint']}")
    ui.markdown("\n".join(bullets)).classes('text-gray-700')

# ---------------- Model status ----------------
if err:
    with ui.card().classes('max-w-4xl mx-auto mt-6 shadow-lg rounded-xl'):
        ui.label('⚠️ Tətbiq açıla bilmədi').classes('text-xl font-semibold text-red-600')
        ui.label(err).classes('text-red-600')
        ui.label('Həll: repoya artifacts/model.pkl yükləyin və ya data/HPLC data.csv əlavə edin ki, server bir dəfə öyrədə bilsin.').classes('text-gray-700')
else:
    model_name, skver, calibrated = detect_model_meta(model, meta)
    with ui.card().classes('max-w-6xl mx-auto mt-4 shadow-lg rounded-xl'):
        ui.label('Model məlumatları').classes('text-lg font-medium')
        with ui.grid(columns=3).classes('gap-4'):
            ui.label(f"Model: {model_name}")
            ui.label(f"scikit-learn versiyası: {skver}")
            ui.label(f"Kalibrasiya: {'Bəli' if calibrated else 'Xeyr'}")

    # ---------------- Giriş formu ----------------
    with ui.card().classes('max-w-6xl mx-auto mt-4 shadow-lg rounded-xl'):
        ui.label('Biomarkerləri daxil et').classes('text-lg font-medium mb-2')

        inputs_num = {}
        inputs_bin = {}
        inputs_cat = {}

        with ui.grid(columns=3).classes('gap-4'):
            # Rəqəmsal: aralıq dropdown + manual input
            for k, metaF in FIELDS.items():
                with ui.column():
                    ui.label(metaF['label']).classes('text-sm')
                    # aralıq seçimləri
                    options = ['—'] + [f"{a}–{b}" for a,b in metaF['bins']]
                    dd = ui.select(options, value='—').props('outlined dense')
                    num = ui.number(
                        label=f"Manual dəyər ( {metaF['min']}–{metaF['max']} )",
                        min=metaF['min'], max=metaF['max'], step=metaF['step'], value=None
                    ).props('outlined dense clearable')
                    ui.icon('info').classes('text-gray-500').tooltip(metaF['hint'])

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

            # Kategoriyalar
            for k,(lab, options) in CATS.items():
                if options:
                    w = ui.select(options, label=lab, value=None, with_input=True).props('outlined dense use-input fill-input')
                else:
                    w = ui.input(label=lab).props('outlined dense')
                inputs_cat[k] = w

        # Presetlər + Clear
        def set_preset(kind):
            presets = {
                'Normal':  {'HbA0':96,'HbA2':2.5,'HbF':0.8,'RBC':4.8,'HB':14,'MCV':88,'MCH':29,'MCHC':34,'RDWcv':12.5,'S_Window':0,'Unknown':0,'Age':30,'Gender':'M','Weekness':'No','Jaundice':'No'},
                'Carrier': {'HbA0':96,'HbA2':3.7,'HbF':1.8,'RBC':5.5,'HB':13,'MCV':74,'MCH':24,'MCHC':33,'RDWcv':14,'S_Window':0,'Unknown':0,'Age':28,'Gender':'F','Weekness':'No','Jaundice':'No'},
                'Disease': {'HbA0':80,'HbA2':2.0,'HbF':8.0,'RBC':5.2,'HB':9.5,'MCV':68,'MCH':22,'MCHC':31,'RDWcv':18,'S_Window':1,'Unknown':0.5,'Age':10,'Gender':'M','Weekness':'Yes','Jaundice':'Yes'},
            }
            data = presets[kind]
            for k2,v in data.items():
                if k2 in inputs_num: inputs_num[k2].value = v
                if k2 in inputs_cat and v in (CATS.get(k2,[None,[]])[1] or []):
                    inputs_cat[k2].value = v
            for k2 in inputs_bin: inputs_bin[k2].value = '—'

        def clear_all():
            for c in inputs_num.values(): c.value = None
            for c in inputs_cat.values(): c.value = None
            for c in inputs_bin.values(): c.value = '—'

        with ui.row().classes('gap-2 mt-2'):
            ui.button('Preset: Normal',  on_click=lambda: set_preset('Normal')).props('flat color=primary')
            ui.button('Preset: Daşıyıcı', on_click=lambda: set_preset('Carrier')).props('flat color=primary')
            ui.button('Preset: Xəstə',    on_click=lambda: set_preset('Disease')).props('flat color=primary')
            ui.button('Təmizlə', on_click=clear_all).props('flat')

        # Qrafikləri göstər gizlət
        show_charts = ui.checkbox('Qrafikləri göstər', value=True)

        # Nəticə bölməsi
        with ui.element('div').classes('mt-4'):
            res_title = ui.label().classes('text-xl font-semibold')
            res_sub   = ui.label().classes('text-sm text-gray-600')

        # 1) Sinif ehtimalları (bar)
        prob_chart = ui.echart({
            'xAxis': {'type':'category', 'data':['Normal','Carrier','Disease']},
            'yAxis': {'type':'value', 'min':0, 'max':1},
            'series':[{'type':'bar','data':[0,0,0]}],
            'grid': {'left': 40, 'right': 10, 'bottom': 30, 'top': 20},
            'tooltip': {'trigger': 'axis'}
        }).classes('w-full h-56')

        # 2) Radar qrafik (seçilmiş göstəricilər)
        radar_keys = ['HbA2','HbF','MCV','MCH','RBC']
        radar_indicators = [{'name':FIELDS[k]['label'], 'max': float(FIELDS[k]['max'])} for k in radar_keys]
        radar_chart = ui.echart({
            'tooltip': {},
            'radar': {'indicator': radar_indicators, 'radius': '65%'},
            'series': [{
                'type': 'radar',
                'data': [{'value': [0]*len(radar_keys), 'name': 'Nümunə'}]
            }],
        }).classes('w-full h-64')

        # 3) Dəyər–aralıq müqayisə (value vs. midpoints)
        cmp_keys = ['HbA2','HbF','MCV','MCH','RDWcv']
        cmp_chart = ui.echart({
            'tooltip': {'trigger': 'axis'},
            'legend': {'data': ['Dəyər', 'Normal orta', 'Daşıyıcı orta', 'Xəstə orta']},
            'xAxis': {'type': 'category', 'data': [FIELDS[k]['label'] for k in cmp_keys]},
            'yAxis': {'type': 'value'},
            'series': [
                {'name': 'Dəyər', 'type': 'bar', 'data': [0]*len(cmp_keys)},
                {'name': 'Normal orta',  'type': 'line', 'data': [0]*len(cmp_keys)},
                {'name': 'Daşıyıcı orta','type': 'line', 'data': [0]*len(cmp_keys)},
                {'name': 'Xəstə orta',   'type': 'line', 'data': [0]*len(cmp_keys)},
            ],
            'grid': {'left': 50, 'right': 10, 'bottom': 40, 'top': 30}
        }).classes('w-full h-64')

        # “Niyə belə?” izahı
        reason_box = ui.label().classes('text-sm mt-2 text-gray-700')

        def update_charts(prob_list, row_vals):
            # ehtimal bar
            prob_chart.options['series'][0]['data'] = prob_list
            prob_chart.update()
            prob_chart.visible = show_charts.value

            # radar
            radar_vals = [row_vals.get(k, 0) or 0 for k in radar_keys]
            radar_chart.options['series'][0]['data'][0]['value'] = radar_vals
            radar_chart.update()
            radar_chart.visible = show_charts.value

            # müqayisə: normal/daşıyıcı/xəstə orta (bins orta nöqtələri)
            def midpoints_for(keys, bin_index):
                mids = []
                for k in keys:
                    a,b = FIELDS[k]['bins'][bin_index]
                    mids.append(round((a+b)/2,3))
                return mids
            # sadə referenslər
            normal_mids  = midpoints_for(cmp_keys, 1 if cmp_keys[0]!='HbA2' else 1)  # orta zona
            carrier_mids = []
            disease_mids = []
            for k in cmp_keys:
                # heuristik: HbA2 üçün “daşıyıcı” = bins[2] orta; HbF üçün yüksəklik bins[2] orta; MCV/MCH üçün “xəstə” = bins[0] orta; RDWcv üçün bins[2] orta
                if k == 'HbA2': 
                    carrier_mids.append(round(sum(FIELDS[k]['bins'][2])/2,3))
                    disease_mids.append(round(sum(FIELDS['MCV']['bins'][0])/2,3))  # uyğun deyil, amma vizual üçün
                elif k == 'HbF':
                    carrier_mids.append(round(sum(FIELDS[k]['bins'][1])/2,3))
                    disease_mids.append(round(sum(FIELDS[k]['bins'][2])/2,3))
                elif k in ['MCV','MCH']:
                    carrier_mids.append(round(sum(FIELDS[k]['bins'][1])/2,3))
                    disease_mids.append(round(sum(FIELDS[k]['bins'][0])/2,3))
                else:  # RDWcv
                    carrier_mids.append(round(sum(FIELDS[k]['bins'][1])/2,3))
                    disease_mids.append(round(sum(FIELDS[k]['bins'][2])/2,3))

            values_now = [row_vals.get(k, 0) or 0 for k in cmp_keys]
            cmp_chart.options['series'][0]['data'] = values_now
            cmp_chart.options['series'][1]['data'] = normal_mids
            cmp_chart.options['series'][2]['data'] = carrier_mids
            cmp_chart.options['series'][3]['data'] = disease_mids
            cmp_chart.update()
            cmp_chart.visible = show_charts.value

        def predict():
            # inputları topla
            row = {}
            for k in FIELDS.keys():
                v = inputs_num[k].value
                row[k] = (None if v=='' else v)
            for k in inputs_cat.keys():
                row[k] = inputs_cat[k].value if inputs_cat[k].value != '' else None

            df = pd.DataFrame([row])

            # diapazon xəbərdarlığı
            msgs = out_of_range_msgs(row)
            if msgs:
                ui.notification('Diapazondan kənar dəyərlər var:\n' + '\n'.join(msgs), type='warning', close_button=True)

            # proqnoz
            yhat = model.predict(df)[0]
            res_title.set_text(f"Nəticə: {LABELS[int(yhat)]}")

            proba = getattr(model, 'predict_proba', None)
            if proba:
                p = proba(df)[0].tolist()
                res_sub.set_text(f"Etibarlılıq: {max(p):.2%}")
                update_charts(p, row)
            else:
                res_sub.set_text('')
                prob_chart.visible = False
                radar_chart.visible = False
                cmp_chart.visible = False

            reason_box.set_text('Niyə belə? ' + rule_based_explanation(row))

        ui.button('PROQNOZ', on_click=predict).props('unelevated color=primary').classes('mt-3')

    # ---------------- Disclaimer ----------------
    with ui.expansion('Məsuliyyət qeydi / Məhdudiyyətlər').classes('max-w-6xl mx-auto mt-4'):
        ui.markdown(
            "- Bu alət **tədqiqat və tədris** məqsədlidir; klinik qərar üçün uyğun deyil.\n"
            "- Dəyərlər laboratoriya referenslərinə görə dəyişə bilər; şübhə olduqda **həkim qərarı** əsasdır.\n"
            "- “Niyə belə?” bölməsi sadə qaydalarla qeyri-formal izah verir; modelin daxili mexanizmini əvəz etmir."
        )

# ---------------- Run ----------------
ui.run(host='0.0.0.0', port=PORT, reload=False, show=False)
