# app.py — HPLC əsaslı Talassemiya Risk Proqnozu (AZ), preset təsvirləri, gözəlləşdirilmiş UI
import os, sys, subprocess, joblib
import pandas as pd
import numpy as np
from nicegui import ui

# Heroku uvicorn "workers" problemi üçün
os.environ["WEB_CONCURRENCY"] = "1"

PORT = int(os.environ.get('PORT', 8080))
MODEL_PATH = 'artifacts/model.pkl'
DATA_PATH = 'data/HPLC data.csv'
LABELS = {0: 'Normal', 1: 'Daşıyıcı', 2: 'Xəstə'}

import re

DASHES = ['–','—','-']  # en/em/hyphen

FILL_DEFAULTS_MODE = 'min'  # alternativ: 'normal_mid'

def to_float_or_none(x):
    if x is None or x == '': return None
    try: return float(x)
    except: return None

def range_mid(v):
    """Dropdown value: None və ya (a,b) gəlir; orta nöqtəni qaytarır."""
    if v is None: return None
    try:
        a,b = v
        return round((float(a)+float(b))/2, 6)
    except:
        return None

def apply_defaults(row):
    """Boş rəqəmsal sahələri default ilə doldur: min və ya normal mid."""
    for k, meta in FIELDS.items():
        if row.get(k) is None:
            if FILL_DEFAULTS_MODE == 'min':
                row[k] = float(meta['min'])
            else:  # 'normal_mid' -> bins[1] orta
                a,b = meta['bins'][1]
                row[k] = round((a+b)/2, 6)
    return row

def out_of_range_msgs(row):
    msgs = []
    EPS = 1e-9
    for k, meta in FIELDS.items():
        v = row.get(k, None)
        if v is None:  # defaults sonra gələcək, burada skip
            continue
        try:
            vf = float(v)
        except:
            continue
        if (vf + EPS) < meta['min'] or (vf - EPS) > meta['max']:
            msgs.append(f"{meta['label']}: {vf}  ({meta['min']}–{meta['max']})")
    return msgs


def parse_range_mid(text):
    """'a–b' / 'a-b' kimi etiketi orta nöqtəyə çevirir; yoxdursa None."""
    if not text or text.strip() == '—':
        return None
    t = text
    for d in DASHES:
        t = t.replace(d, '-')
    nums = re.findall(r'[-+]?\d*\.?\d+', t)
    if len(nums) >= 2:
        a = float(nums[0]); b = float(nums[1])
        return round((a + b) / 2, 6)
    return None

def to_float_or_none(x):
    """Manual+dropdown qarışıq dəyərləri təhlükəsiz float-a çevirir."""
    if x is None or x == '':
        return None
    try:
        return float(x)
    except Exception:
        return None


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

# ---------------- UI stil (yumşaq fon, kart kölgəsi, hero gradient) ----------------
ui.add_head_html("""
<style>
  body { background: #f6f7fb; }
  .app-card { border-radius: 16px; box-shadow: 0 10px 24px rgba(23,43,77,0.07); }
  .app-header { background: linear-gradient(90deg, #2563eb 0%, #4f46e5 50%, #7c3aed 100%); }
  .section-title { font-weight: 700; font-size: 1.1rem; color: #1f2937; }
  .muted { color: #6b7280; }
  .chip { display:inline-flex; align-items:center; gap:.4rem; background:#eef2ff; color:#3730a3;
          padding:.25rem .6rem; border-radius:999px; font-size:.85rem; }
</style>
""")

# ---------------- Sahələr, aralıqlar və izahlar ----------------
FIELDS = {
    'HbA0':  {'label':'HbA0 (%)',          'min':0,  'max':100, 'step':0.1,
              'bins': [(0,90),(90,97),(97,100)], 'hint':'Əsas hemoglobin fraksiyası (HPLC).'},
    'HbA2':  {'label':'HbA2 (%)',          'min':0,  'max':10,  'step':0.1,
              'bins': [(0,2.0),(2.0,3.5),(3.5,10)], 'hint':'Adətən 1.5–3.5%. >3.5% daşıyıcılıq əlaməti ola bilər.'},
    'HbF':   {'label':'HbF (%)',           'min':0,  'max':40,  'step':0.1,
              'bins': [(0,2.0),(2.0,10),(10,40)],  'hint':'Fetal Hb; yüksəkliyi talassemiya ilə uyğun ola bilər.'},
    'RBC':   {'label':'RBC (10^12/L)',     'min':1,  'max':8,   'step':0.01,
              'bins': [(1,4.5),(4.5,5.5),(5.5,8)], 'hint':'Eritrosit sayı; talassemiyada bəzən yüksək/normal.'},
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

# ---------------- Preset dəyərləri və təsvirləri ----------------
PRESETS = {
    'Normal': {
        'HbA0':96,'HbA2':2.3,'HbF':0.8,'RBC':4.9,'HB':13.8,'MCV':88,'MCH':29,'MCHC':33.5,'RDWcv':12.4,
        'S_Window':0,'Unknown':0,'Age':30,'Gender':'M','Weekness':'No','Jaundice':'No'
    },
    'Carrier': {
        'HbA0': 95.0,
        'HbA2': 5.5,   # ↑ daha kəskin marker
        'HbF':  1.5,
        'RBC':  6.0,   # tez-tez yüksək/normal
        'HB':   12.2,
        'MCV':  66,    # ↓ mikrositoz
        'MCH':  21,    # ↓ hipoxromiya
        'MCHC': 32.0,
        'RDWcv': 15.5,
        'S_Window': 0,
        'Unknown': 0,
        'Age': 24,
        'Gender': 'F',
        'Weekness': 'No',
        'Jaundice': 'No'
    },

    'Disease': {
        'HbA0':60,'HbA2':2.5,'HbF':40,'RBC':3.2,'HB':7.5,'MCV':65,'MCH':19,'MCHC':31,'RDWcv':18.5,
        'S_Window':1.0,'Unknown':0.5,'Age':9,'Gender':'M','Weekness':'Yes','Jaundice':'Yes'
    }
}
PRESET_INFO = {
    'Normal': (
        "### Normal profil\n"
        "- HbA2: ~1.5–3.5%\n"
        "- HbF: <2%\n"
        "- RBC / MCV / MCH: normal intervalda\n"
        "- Klinik baxımdan sağlamlıq göstəriciləri uyğundur"
    ),
    'Carrier': (
        "### Daşıyıcı (β-thal trait) profil\n"
        "- **HbA2: >3.5%** (əsas marker)\n"
        "- MCV aşağı (mikrositoz), MCH aşağı (hipoxromiya)\n"
        "- RBC çox vaxt normaldan yüksək/normal\n"
        "- Adətən yüngül və ya simptomsuz gediş"
    ),
    'Disease': (
        "### Xəstə (β-thal major/intermedia) profil\n"
        "- **HbF: çox yüksək** (tez-tez >10–20%, ağır hallarda daha da yüksək)\n"
        "- Hb: aşağı (anemiya), RBC: aşağı\n"
        "- MCV / MCH aşağı, RDWcv yüksələ bilər\n"
        "- Klinik nəzarət və hematoloji qiymətləndirmə tələb olunur"
    ),
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
    EPS = 1e-9
    for k, meta in FIELDS.items():
        v = row.get(k, None)
        if v is None:
            continue  # boş sahələrə xəbərdarlıq etmə
        try:
            vf = float(v)
        except Exception:
            continue
        # Float yuvarlaqlaşma səhvlərinə elastik davran
        if (vf + EPS) < meta['min'] or (vf - EPS) > meta['max']:
            msgs.append(f"{meta['label']}: {vf}  ({meta['min']}–{meta['max']})")
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
with ui.header().classes('app-header text-white'):
    with ui.row().classes('w-full items-center justify-between'):
        with ui.column().classes('py-3'):
            ui.label('Talassemiya Risk Proqnozu (HPLC göstəriciləri)').classes('text-2xl font-bold')
            ui.label('Demo • Klinik istifadə üçün deyil').classes('opacity-80')
        with ui.row().classes('items-center gap-2'):
            ui.html('<span class="chip">HPLC</span>')
            ui.html('<span class="chip">ML Model</span>')
            ui.html('<span class="chip">Education/Research</span>')

# ---------------- Yuxarı məlumat kartı (layihə + dəyişənlər) ----------------
with ui.card().classes('app-card max-w-6xl mx-auto mt-6'):
    with ui.row().classes('items-center gap-2'):
        ui.icon('analytics').classes('text-indigo-600')
        ui.label('Layihə haqqında qısa məlumat').classes('section-title')
    ui.markdown(
        "**Məqsəd:** HPLC və qan göstəricilərinə əsasən talassemiya statusunun "
        "(Normal / Daşıyıcı / Xəstə) proqnozlaşdırılması.\n\n"
        "**Qeyd:** Bu alət tədqiqat və tədris məqsədlidir; klinik qərar üçün uyğun deyil."
    ).classes('muted')
    ui.separator()
    with ui.row().classes('items-center gap-2 mt-2'):
        ui.icon('science').classes('text-emerald-600')
        ui.label('Göstəricilər və mənaları').classes('section-title')
    bullets = [f"- **{meta['label']}**: {meta['hint']}" for k, meta in FIELDS.items()]
    ui.markdown("\n".join(bullets)).classes('muted')

# ---------------- Model status ----------------
if err:
    with ui.card().classes('app-card max-w-4xl mx-auto mt-6'):
        ui.label('⚠️ Tətbiq açıla bilmədi').classes('text-xl font-semibold text-red-600')
        ui.label(err).classes('text-red-600')
        ui.label('Həll: repoya artifacts/model.pkl yükləyin və ya data/HPLC data.csv əlavə edin ki, server bir dəfə öyrədə bilsin.').classes('muted')
else:
    model_name, skver, calibrated = detect_model_meta(model, meta)
    with ui.card().classes('app-card max-w-6xl mx-auto mt-4'):
        ui.label('Model məlumatları').classes('section-title')
        with ui.grid(columns=3).classes('gap-4'):
            ui.label(f"Model: {model_name}")
            ui.label(f"scikit-learn versiyası: {skver}")
            ui.label(f"Kalibrasiya: {'Bəli' if calibrated else 'Xeyr'}")

    # ---------------- Giriş formu ----------------
    with ui.card().classes('app-card max-w-6xl mx-auto mt-4'):
        ui.label('Biomarkerləri daxil et').classes('section-title')

        inputs_num, inputs_bin, inputs_cat = {}, {}, {}

        with ui.grid(columns=3).classes('gap-4'):
            # Rəqəmsal: aralıq dropdown + manual input
            for k, metaF in FIELDS.items():
                with ui.column():
                    ui.label(metaF['label']).classes('text-sm')

                    # label -> value (value: None və ya (a,b))
                    range_options = {'—': None}
                    for a,b in metaF['bins']:
                        label = f"{a}-{b}"        # sadə '-' yazırıq ki, unicode tire problemi olmasın
                        range_options[label] = (a,b)
                    dd = ui.select(range_options, value=None).props('outlined dense')

                    num = ui.number(
                        label=f"Manual dəyər ( {metaF['min']}–{metaF['max']} )",
                        min=metaF['min'], max=metaF['max'], step=metaF['step'], value=None
                    ).props('outlined dense clearable')
                    ui.icon('info').classes('text-gray-500').tooltip(metaF['hint'])

                    def on_dd_change(e, key=k, meta=metaF, box=num):
                        mid = range_mid(e.value)  # e.value artıq None və ya (a,b)-dir
                        box.value = mid
                    dd.on('update:model-value', on_dd_change)


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
            data = PRESETS[kind]
            for k2, v in data.items():
                if k2 in inputs_num: inputs_num[k2].value = v
                if k2 in inputs_cat and v in (CATS.get(k2, [None, []])[1] or []):
                    inputs_cat[k2].value = v
            for k2 in inputs_bin: inputs_bin[k2].value = None
            preset_info_box.set_content(PRESET_INFO.get(kind, ''))

        def clear_all():
            for c in inputs_num.values(): c.value = None
            for c in inputs_cat.values(): c.value = None
            for c in inputs_bin.values(): c.value = None  # ← IMPORTANT: '—' yox, None
            preset_info_box.set_content('')
            res_title.set_text(''); res_sub.set_text(''); reason_box.set_text('')
            prob_chart.options['series'][0]['data'] = [0,0,0]; prob_chart.update()
            radar_chart.options['series'][0]['data'][0]['value'] = [0]*len(radar_keys); radar_chart.update()
            cmp_chart.options['series'][0]['data'] = [None]*len(cmp_keys)
            cmp_chart.options['series'][1]['data'] = [None]*len(cmp_keys)
            cmp_chart.options['series'][2]['data'] = [None]*len(cmp_keys)
            cmp_chart.options['series'][3]['data'] = [None]*len(cmp_keys)
            cmp_chart.update()



        with ui.row().classes('gap-2 mt-2'):
            ui.button('Preset: Normal',    on_click=lambda: set_preset('Normal')).props('unelevated color=primary').classes('rounded-lg')
            ui.button('Preset: Daşıyıcı',  on_click=lambda: set_preset('Carrier')).props('unelevated color=primary').classes('rounded-lg')
            ui.button('Preset: Xəstə',     on_click=lambda: set_preset('Disease')).props('unelevated color=primary').classes('rounded-lg')
            ui.button('Təmizlə',           on_click=clear_all).props('outline color=grey-7').classes('rounded-lg')

        ui.separator()
        ui.label('Seçilən preset təsviri').classes('section-title mt-2')
        preset_info_box = ui.markdown('')  # preset təsviri burada göstərilir

        # Qrafikləri göstər gizlət
        show_charts = ui.checkbox('Qrafikləri göstər', value=True)

        # Nəticə header
        with ui.element('div').classes('mt-4'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('insights').classes('text-amber-600')
                ui.label('Proqnoz və ehtimallar').classes('section-title')
            res_title = ui.label().classes('text-xl font-semibold')
            res_sub   = ui.label().classes('text-sm muted')

        # 1) Sinif ehtimalları (bar)
        prob_chart = ui.echart({
            'xAxis': {'type':'category', 'data':['Normal','Daşıyıcı','Xəstə']},
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


        def safe_nonneg(v):
            try:
                x = float(v)
                return None if x < 0 else x  # <0 isə qrafikdə çəkməməyi seçirik
            except Exception:
                return None
        
        def update_charts(prob_list, row_vals):
            # ehtimal bar (0..1 clamp)
            probs = []
            for x in prob_list:
                try:
                    fx = float(x)
                except Exception:
                    fx = 0.0
                probs.append(max(0.0, min(1.0, fx)))
            prob_chart.options['yAxis']['min'] = 0
            prob_chart.options['yAxis']['max'] = 1
            prob_chart.options['series'][0]['data'] = probs
            prob_chart.update()
            prob_chart.visible = show_charts.value
        
            # radar
            radar_vals = [to_float_or_none(row_vals.get(k, None)) or 0.0 for k in radar_keys]
            radar_chart.options['series'][0]['data'][0]['value'] = radar_vals
            radar_chart.update()
            radar_chart.visible = show_charts.value
        
            # value vs midpoints (negativləri None elə ki, aşağı bar olmasın)
            def midpoints_for(keys, bin_index):
                mids = []
                for k in keys:
                    a, b = FIELDS[k]['bins'][bin_index]
                    mids.append(round((a + b) / 2, 6))
                return mids
        
            normal_mids  = midpoints_for(cmp_keys, 1)
            carrier_mids, disease_mids = [], []
            for k in cmp_keys:
                if k == 'HbA2':
                    carrier_mids.append(round(sum(FIELDS[k]['bins'][2]) / 2, 6))
                    disease_mids.append(round(sum(FIELDS[k]['bins'][1]) / 2, 6))
                elif k == 'HbF':
                    carrier_mids.append(round(sum(FIELDS[k]['bins'][1]) / 2, 6))
                    disease_mids.append(round(sum(FIELDS[k]['bins'][2]) / 2, 6))
                elif k in ['MCV','MCH']:
                    carrier_mids.append(round(sum(FIELDS[k]['bins'][1]) / 2, 6))
                    disease_mids.append(round(sum(FIELDS[k]['bins'][0]) / 2, 6))
                else:  # RDWcv
                    carrier_mids.append(round(sum(FIELDS[k]['bins'][1]) / 2, 6))
                    disease_mids.append(round(sum(FIELDS[k]['bins'][2]) / 2, 6))
        
            values_now = [safe_nonneg(row_vals.get(k, None)) for k in cmp_keys]
        
            cmp_chart.options['xAxis']['data'] = [FIELDS[k]['label'] for k in cmp_keys]
            cmp_chart.options['yAxis']['min'] = 0  # aşağı bar olmasın
            cmp_chart.options['series'][0]['data'] = values_now
            cmp_chart.options['series'][1]['data'] = normal_mids
            cmp_chart.options['series'][2]['data'] = carrier_mids
            cmp_chart.options['series'][3]['data'] = disease_mids
            cmp_chart.update()
            cmp_chart.visible = show_charts.value


        def predict():
            # inputları topla: manual üstünlük, sonra dropdown orta nöqtə
            row = {}
            for k in FIELDS.keys():
                manual = to_float_or_none(inputs_num[k].value)
                if manual is None:
                    mid = range_mid(inputs_bin[k].value)  # None və ya orta nöqtə
                    row[k] = to_float_or_none(mid)
                else:
                    row[k] = manual
            
            for k in inputs_cat.keys():
                v = inputs_cat[k].value
                row[k] = v if v not in ('', None) else None
            
            # boş rəqəmsallara DEFAULT doldur (sənin istəyinlə: minimal dəyər)
            row = apply_defaults(row)
            
            # indi xəbərdarlığı et (artıq yalnış flag almamalıdır)
            msgs = out_of_range_msgs(row)
            if msgs:
                ui.notification('Diapazondan kənar dəyərlər:\n' + '\n'.join(msgs), type='warning', close_button=True)
            
            df = pd.DataFrame([row])


            # diapazon xəbərdarlığı
            msgs = out_of_range_msgs(row)
            if msgs:
                ui.notification('Diapazondan kənar dəyərlər var:\n' + '\n'.join(msgs), type='warning', close_button=True)

            # proqnoz
            yhat = model.predict(df)[0]
            proba = getattr(model, 'predict_proba', None)
            
            # proqnoz (model) + klinik qayda (override)
            row_vals = {k: df.iloc[0].get(k, None) for k in df.columns}
            hbA2 = row_vals.get('HbA2', None)
            mcv  = row_vals.get('MCV', None)
            hb   = row_vals.get('HB', None)
            hbf  = row_vals.get('HbF', None)
            
            # 1) Əvvəl xəstə üçün sərt qayda (HbF çox yüksək + Hb aşağı)
            if (hbf is not None and hbf >= 12.0) and (hb is not None and hb <= 10.5):
                yhat = 2
                p = [0.05, 0.10, 0.85]  # vizual üçün ehtimal bölgüsü
                res_title.set_text(f"Nəticə: {LABELS[yhat]}")
                res_sub.set_text(f"Etibarlılıq: {max(p):.2%}")
                update_charts(p, row_vals)
                reason_box.set_text('Niyə belə? HbF yüksək və Hb aşağı → Xəstə (override)')
            else:
                # 2) Daşıyıcı üçün SƏRT qayda: HbA2 ≥ 3.8% və MCV ≤ 80 → DAŞIYICI
                if (hbA2 is not None and hbA2 >= 3.8) and (mcv is not None and mcv <= 80):
                    yhat = 1
                    p = [0.10, 0.80, 0.10]  # vizual üçün ehtimal bölgüsü
                    res_title.set_text(f"Nəticə: {LABELS[yhat]}")
                    res_sub.set_text(f"Etibarlılıq: {max(p):.2%}")
                    update_charts(p, row_vals)
                    reason_box.set_text('Niyə belə? HbA2 ≥ 3.8% və MCV ≤ 80 → Daşıyıcı (override)')
                else:
                    # 3) Heç biri deyilsə, modelə burax
                    yhat = int(model.predict(df)[0])
                    proba = getattr(model, 'predict_proba', None)
                    if proba:
                        p = proba(df)[0].astype(float)
                        p = np.clip(p, 1e-6, None); p = (p / p.sum()).tolist()
                        res_title.set_text(f"Nəticə: {LABELS[int(np.argmax(p))]}")
                        res_sub.set_text(f"Etibarlılıq: {max(p):.2%}")
                        update_charts(p, row_vals)
                    else:
                        res_title.set_text(f"Nəticə: {LABELS[yhat]}")
                        res_sub.set_text('')
                        prob_chart.visible = False
                        radar_chart.visible = False
                        cmp_chart.visible = False
                    reason_box.set_text('Niyə belə? ' + rule_based_explanation(row_vals))




        ui.button('PROQNOZ', on_click=predict).props('unelevated color=primary size=lg').classes('mt-3 rounded-xl')

    # ---------------- Disclaimer ----------------
    with ui.expansion('Məsuliyyət qeydi / Məhdudiyyətlər').classes('app-card max-w-6xl mx-auto mt-4'):
        ui.markdown(
            "- Bu alət **tədqiqat və tədris** məqsədlidir; klinik qərar üçün uyğun deyil.\n"
            "- Dəyərlər laboratoriya referenslərinə görə dəyişə bilər; şübhə olduqda **həkim qərarı** əsasdır.\n"
            "- “Niyə belə?” bölməsi sadə qaydalarla qeyri-formal izah verir; modelin daxili mexanizmini əvəz etmir."
        )

# ---------------- Run ----------------
ui.run(host='0.0.0.0', port=PORT, reload=False, show=False)





