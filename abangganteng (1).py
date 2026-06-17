"""
Streamlit App — Regresi Rating Produk Skincare Sociola
Tugas Besar Machine Learning
Dataset: https://raw.githubusercontent.com/RufusLubis/Dataset/refs/heads/main/DatasetSociola.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sociola — Regresi Rating Skincare",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hero Banner ── */
    .hero {
        background: linear-gradient(135deg, #c0392b 0%, #8e1010 60%, #2c0a0a 100%);
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        color: white;
        box-shadow: 0 8px 32px rgba(192,57,43,0.25);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "🧴";
        position: absolute;
        right: 32px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 6rem;
        opacity: 0.15;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 0.95rem;
        opacity: 0.85;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 10px;
        letter-spacing: 0.5px;
    }

    /* ── Metric Cards ── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #fff;
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        border: 1px solid #f0f0f0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #C0392B;
        line-height: 1;
    }
    .kpi-icon {
        font-size: 1.1rem;
        margin-bottom: 4px;
    }

    /* ── Section Headers ── */
    .sec-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0 0 12px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sec-header::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 20px;
        background: linear-gradient(180deg, #C0392B, #E74C3C);
        border-radius: 2px;
    }

    /* ── Info / Success / Warning Boxes ── */
    .box-info {
        background: linear-gradient(135deg, #EBF5FB, #D6EAF8);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        border-left: 4px solid #2E86C1;
        font-size: 0.9rem;
    }
    .box-success {
        background: linear-gradient(135deg, #EAFAF1, #D5F5E3);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        border-left: 4px solid #27AE60;
        font-size: 0.9rem;
    }
    .box-warning {
        background: linear-gradient(135deg, #FEF9E7, #FDEBD0);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        border-left: 4px solid #E67E22;
        font-size: 0.9rem;
    }
    .box-error {
        background: linear-gradient(135deg, #FDEDEC, #FADBD8);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        border-left: 4px solid #C0392B;
        font-size: 0.9rem;
    }

    /* ── Prediction Result Card ── */
    .pred-card {
        background: linear-gradient(135deg, #EAFAF1, #D5F5E3);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #A9DFBF;
        box-shadow: 0 4px 20px rgba(39,174,96,0.12);
    }
    .pred-label {
        font-size: 0.9rem;
        color: #555;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .pred-value {
        font-size: 4rem;
        font-weight: 800;
        color: #1E8449;
        line-height: 1;
        margin: 8px 0;
    }
    .pred-stars {
        font-size: 1.4rem;
        margin: 4px 0 8px 0;
    }
    .pred-brand {
        font-size: 0.82rem;
        color: #888;
    }

    /* ── Loading Overlay ── */
    .loading-text {
        text-align: center;
        color: #C0392B;
        font-weight: 600;
        font-size: 1rem;
        padding: 20px;
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8f9fa;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.88rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0a0a 0%, #2c1515 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f0e6e6 !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div {
        background: rgba(192,57,43,0.3) !important;
    }

    /* ── Divider ── */
    .fancy-divider {
        height: 2px;
        background: linear-gradient(90deg, #C0392B, transparent);
        border: none;
        border-radius: 2px;
        margin: 20px 0;
    }

    /* ── Model Badge ── */
    .model-badge {
        display: inline-block;
        background: #C0392B;
        color: white !important;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_URL  = "https://raw.githubusercontent.com/RufusLubis/Dataset/refs/heads/main/DatasetSociola.csv"
FEATURE_COLS = [
    'harga_jual_avg', 'harga_asli_avg', 'pct_diskon',
    'selisih_harga',  'rasio_diskon',   'jumlah_size',
    'nama_len',       'brand_count',
]

# ── Matplotlib Theme ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'axes.facecolor'   : '#FAFAFA',
    'figure.facecolor' : '#FFFFFF',
    'axes.titleweight' : 'bold',
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 10,
})

# ── Helper Functions ──────────────────────────────────────────────────────────
def parse_harga(x):
    if pd.isna(x): return np.nan
    nums = re.findall(r'[\d.]+', str(x))
    nums = [float(n.replace('.', '')) for n in nums if len(n) > 2]
    return np.mean(nums) if nums else np.nan

def parse_pct(x):
    if pd.isna(x): return 0.0
    m = re.search(r'(\d+)', str(x))
    return float(m.group(1)) if m else 0.0

def preprocess(df):
    data = df.copy()
    data['harga_jual_avg'] = data['harga_jual'].apply(parse_harga)
    data['harga_asli_avg'] = data['harga_asli'].apply(parse_harga)
    data['pct_diskon']     = data['persentase_diskon'].apply(parse_pct)
    data['selisih_harga']  = data['harga_asli_avg'] - data['harga_jual_avg']
    data['rasio_diskon']   = data['harga_jual_avg'] / data['harga_asli_avg'].replace(0, np.nan)
    data['nama_len']       = data['nama_produk'].str.len()
    data['brand_count']    = data['brand'].map(data['brand'].value_counts())
    data = data.dropna(subset=['rating']).reset_index(drop=True)
    for col in ['harga_jual_avg', 'harga_asli_avg', 'selisih_harga', 'rasio_diskon']:
        data[col] = data[col].fillna(data[col].median())
    return data

@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATASET_URL)

def stars(rating):
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">🧴 Regresi Rating Produk Skincare Sociola</p>
    <p class="hero-sub">Supervised Machine Learning — Memprediksi rating produk skincare berdasarkan fitur harga, brand, dan diskon</p>
    <span class="hero-badge">📚 Tugas Besar Machine Learning</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")
    st.markdown("---")

    st.markdown("""
    <div style="background:rgba(192,57,43,0.2); border-radius:10px; padding:12px; margin-bottom:16px; border:1px solid rgba(192,57,43,0.3);">
        <div style="font-size:0.8rem; font-weight:600; margin-bottom:4px;">📦 Sumber Dataset</div>
        <div style="font-size:0.72rem; word-break:break-all; opacity:0.85;">GitHub · RufusLubis/Dataset</div>
        <div style="font-size:0.7rem; opacity:0.6; margin-top:4px;">Auto-loaded · Tidak perlu upload</div>
    </div>
    """, unsafe_allow_html=True)

    test_size = st.slider("📐 Ukuran Test Set (%)", 10, 40, 20, help="Proporsi data yang dipakai untuk evaluasi") / 100

    st.markdown("---")
    run_tuning = st.checkbox(
        "🔧 Hyperparameter Tuning",
        value=False,
        help="GridSearchCV pada Gradient Boosting — memakan waktu lebih lama"
    )

    st.markdown("---")
    st.markdown("**🤖 Model yang Dilatih:**")
    models_list = [
        ("📈", "Linear Regression"),
        ("🔵", "Ridge Regression"),
        ("🌳", "Decision Tree"),
        ("🌲", "Random Forest"),
        ("⚡", "Gradient Boosting"),
        ("🔍", "K-Nearest Neighbors"),
    ]
    for icon, name in models_list:
        st.markdown(f"<div style='font-size:0.82rem; padding:2px 0;'>{icon} {name}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem; opacity:0.5; text-align:center;'>Sociola ML App v2.0</div>", unsafe_allow_html=True)

# ── Load Dataset ──────────────────────────────────────────────────────────────
with st.spinner("⏳ Memuat dataset dari GitHub..."):
    try:
        df_raw = load_dataset()
        data   = preprocess(df_raw)
        load_ok = True
    except Exception as e:
        load_ok = False
        st.markdown(f"""
        <div class="box-error">
        ❌ <b>Gagal memuat dataset.</b><br>
        Pastikan koneksi internet tersedia.<br>
        Error: <code>{e}</code>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

if load_ok:
    st.markdown(f"""
    <div class="box-success" style="margin-bottom:20px;">
    ✅ <b>Dataset berhasil dimuat otomatis dari GitHub.</b>
    Ditemukan <b>{len(df_raw):,}</b> baris · <b>{df_raw.shape[1]}</b> kolom · <b>{df_raw['brand'].nunique()}</b> brand unik
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Data & EDA",
    "🔬  Feature Engineering",
    "🤖  Training Model",
    "🏆  Evaluasi",
    "🔮  Prediksi Manual",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data & EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # KPI Cards
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-icon">📋</div>
            <div class="kpi-label">Total Produk</div>
            <div class="kpi-value">{len(df_raw):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">✅</div>
            <div class="kpi-label">Data Valid</div>
            <div class="kpi-value">{len(data):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">🏷️</div>
            <div class="kpi-label">Brand Unik</div>
            <div class="kpi-value">{df_raw['brand'].nunique():,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-icon">⭐</div>
            <div class="kpi-label">Rata-rata Rating</div>
            <div class="kpi-value">{data['rating'].mean():.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔎 Lihat Sample Data (10 Baris Pertama)"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    with st.expander("📐 Statistik Deskriptif"):
        st.dataframe(df_raw.describe(include='all').T.round(3), use_container_width=True)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # Distribusi Rating
    st.markdown('<p class="sec-header">Distribusi Target — Rating Produk</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(7, 4))
        n, bins, patches = ax.hist(data['rating'], bins=20, color='#C0392B', edgecolor='white', alpha=0.85)
        ax.axvline(data['rating'].mean(),   color='#2E86C1', linestyle='--', lw=2, label=f"Mean = {data['rating'].mean():.2f}")
        ax.axvline(data['rating'].median(), color='#F39C12', linestyle='--', lw=2, label=f"Median = {data['rating'].median():.2f}")
        ax.set_title('Distribusi Rating Produk')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frekuensi')
        ax.legend(framealpha=0.8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(7, 4))
        vc = data['rating'].value_counts().sort_index()
        bars = ax.bar(vc.index.astype(str), vc.values, color='#E74C3C', edgecolor='white', alpha=0.85)
        ax.bar_label(bars, fmt='%d', fontsize=8, padding=3, color='#333')
        ax.set_title('Frekuensi per Nilai Rating')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Jumlah Produk')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    col_c, col_d, col_e = st.columns(3)
    col_c.metric("📊 Rating Mean",   f"{data['rating'].mean():.3f}")
    col_d.metric("📉 Std Deviasi",   f"{data['rating'].std():.3f}")
    col_e.metric("📈 Rating Maks",   f"{data['rating'].max():.1f}")

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # Top Brands
    st.markdown('<p class="sec-header">Top 15 Brand — Jumlah Produk & Avg Rating</p>', unsafe_allow_html=True)

    col_f, col_g = st.columns(2)

    with col_f:
        top15 = df_raw['brand'].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_bar = plt.cm.Reds_r(np.linspace(0.2, 0.8, len(top15)))
        top15.sort_values().plot(kind='barh', color=colors_bar, ax=ax, edgecolor='white')
        ax.set_title('Top 15 Brand — Jumlah Produk')
        ax.set_xlabel('Jumlah Produk')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_g:
        brand_r = (data.groupby('brand')['rating']
                   .agg(['mean', 'count'])
                   .query('count >= 5')
                   .sort_values('mean'))
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_bar2 = plt.cm.Greens(np.linspace(0.3, 0.9, 10))
        brand_r.tail(10)['mean'].plot(kind='barh', color=colors_bar2, ax=ax, edgecolor='white')
        ax.set_title('Top 10 Brand — Avg Rating (min 5 produk)')
        ax.set_xlabel('Avg Rating')
        ax.set_xlim([brand_r['mean'].min() * 0.98, brand_r['mean'].max() * 1.01])
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # Heatmap Korelasi
    st.markdown('<p class="sec-header">Korelasi Fitur Numerik vs Rating</p>', unsafe_allow_html=True)

    num_cols = ['harga_jual_avg', 'harga_asli_avg', 'pct_diskon',
                'jumlah_size', 'nama_len', 'selisih_harga', 'rasio_diskon', 'rating']
    corr = data[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
        ax=ax, linewidths=0.5, linecolor='white',
        annot_kws={"size": 9}, square=True
    )
    ax.set_title('Heatmap Korelasi Fitur Numerik', pad=14)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

    # Scatter Plots
    st.markdown('<p class="sec-header">Scatter — Rating vs Harga & Diskon</p>', unsafe_allow_html=True)

    col_h, col_i = st.columns(2)
    with col_h:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(data['harga_jual_avg'], data['rating'], alpha=0.25, color='#C0392B', s=18)
        ax.set_xlabel('Harga Jual Avg (Rp)')
        ax.set_ylabel('Rating')
        ax.set_title('Rating vs Harga Jual')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_i:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(data['pct_diskon'], data['rating'], alpha=0.25, color='#E67E22', s=18)
        ax.set_xlabel('Persentase Diskon (%)')
        ax.set_ylabel('Rating')
        ax.set_title('Rating vs Diskon')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="sec-header">Fitur yang Dibangun dari Kolom Asli</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="box-info">
    <b>📐 8 Fitur Numerik Terrekayasa:</b><br><br>
    &nbsp;&nbsp;🔹 <code>harga_jual_avg</code>    — Rata-rata harga jual (min-max) setelah diskon<br>
    &nbsp;&nbsp;🔹 <code>harga_asli_avg</code>    — Rata-rata harga asli sebelum diskon<br>
    &nbsp;&nbsp;🔹 <code>pct_diskon</code>        — Persentase diskon dalam bentuk numerik<br>
    &nbsp;&nbsp;🔹 <code>selisih_harga</code>     — Nominal penghematan (harga asli – harga jual)<br>
    &nbsp;&nbsp;🔹 <code>rasio_diskon</code>      — Rasio harga jual terhadap harga asli<br>
    &nbsp;&nbsp;🔹 <code>nama_len</code>          — Jumlah karakter nama produk<br>
    &nbsp;&nbsp;🔹 <code>brand_count</code>       — Jumlah produk per brand (popularitas brand)<br>
    &nbsp;&nbsp;🔹 <code>brand_avg_rating</code>  — Target Encoding brand (dari train set, anti data leakage)<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
    st.markdown('<p class="sec-header">Mutual Information — Relevansi Fitur vs Rating</p>', unsafe_allow_html=True)

    X_temp = data[FEATURE_COLS].copy()
    y_temp = data['rating']
    with st.spinner("Menghitung Mutual Information..."):
        mi = mutual_info_regression(StandardScaler().fit_transform(X_temp), y_temp, random_state=42)
    mi_s = pd.Series(mi, index=FEATURE_COLS).sort_values()

    col_mi_a, col_mi_b = st.columns([3, 1])

    with col_mi_a:
        fig, ax = plt.subplots(figsize=(9, 4))
        colors_mi = plt.cm.Purples(np.linspace(0.35, 0.9, len(mi_s)))
        mi_s.plot(kind='barh', color=colors_mi, ax=ax, edgecolor='white')
        ax.bar_label(ax.containers[0], fmt='%.4f', fontsize=8.5, padding=4)
        ax.set_title('Mutual Information Tiap Fitur vs Rating')
        ax.set_xlabel('Mutual Information Score')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_mi_b:
        st.markdown("**📊 Ranking Fitur**")
        mi_df = (mi_s.sort_values(ascending=False)
                 .rename("MI Score").reset_index()
                 .rename(columns={'index': 'Fitur'}).round(4))
        st.dataframe(mi_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="box-success">
    ✅ Semua 8 fitur numerik dipertahankan + <b>brand_avg_rating</b> (target encoding).<br>
    🛡️ Target encoding dilakukan <b>SETELAH split</b> untuk mencegah data leakage.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Training Model
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="sec-header">Training & Perbandingan 6 Model Regresi</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="box-info">
    ⚙️ Konfigurasi aktif: <b>Test size = {int(test_size*100)}%</b> &nbsp;|&nbsp;
    <b>Tuning = {'✅ Aktif' if run_tuning else '❌ Non-aktif'}</b>
    </div>
    """, unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("▶️ Jalankan Training", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("🔄 Training 6 model regresi... harap tunggu"):

            X = data[FEATURE_COLS + ['brand']].copy()
            y = data['rating']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Target Encoding (train only — anti leakage)
            global_mean     = y_train.mean()
            brand_mean_tr   = y_train.groupby(X_train['brand']).mean()
            X_train['brand_avg_rating'] = X_train['brand'].map(brand_mean_tr).fillna(global_mean)
            X_test['brand_avg_rating']  = X_test['brand'].map(brand_mean_tr).fillna(global_mean)
            X_train = X_train.drop(columns=['brand'])
            X_test  = X_test.drop(columns=['brand'])
            all_features = FEATURE_COLS + ['brand_avg_rating']

            scaler = StandardScaler()
            Xtr_s  = scaler.fit_transform(X_train)
            Xte_s  = scaler.transform(X_test)

            models_dict = {
                'Linear Regression' : LinearRegression(),
                'Ridge Regression'  : Ridge(alpha=1.0),
                'Decision Tree'     : DecisionTreeRegressor(random_state=42),
                'Random Forest'     : RandomForestRegressor(n_estimators=200, random_state=42),
                'Gradient Boosting' : GradientBoostingRegressor(random_state=42),
                'KNN'               : KNeighborsRegressor(),
            }

            rows, trained = [], {}
            prog = st.progress(0, text="Melatih model...")
            for i, (name, model) in enumerate(models_dict.items()):
                prog.progress((i + 1) / len(models_dict), text=f"Training: {name}")
                model.fit(Xtr_s, y_train)
                p_te = model.predict(Xte_s)
                p_tr = model.predict(Xtr_s)
                rows.append({
                    'Model'      : name,
                    'MAE Test'   : mean_absolute_error(y_test, p_te),
                    'RMSE Test'  : np.sqrt(mean_squared_error(y_test, p_te)),
                    'R² Test'    : r2_score(y_test, p_te),
                    'MAE Train'  : mean_absolute_error(y_train, p_tr),
                    'RMSE Train' : np.sqrt(mean_squared_error(y_train, p_tr)),
                    'R² Train'   : r2_score(y_train, p_tr),
                })
                trained[name] = model
            prog.empty()

            result = pd.DataFrame(rows).sort_values('MAE Test').reset_index(drop=True)

            # Hyperparameter Tuning (opsional)
            best_params = None
            if run_tuning:
                st.info("🔧 Menjalankan GridSearchCV pada Gradient Boosting...")
                param_grid = {
                    'n_estimators' : [100, 200],
                    'max_depth'    : [2, 3, 4],
                    'learning_rate': [0.05, 0.1],
                    'subsample'    : [0.8, 1.0],
                }
                grid = GridSearchCV(
                    GradientBoostingRegressor(random_state=42), param_grid,
                    scoring='neg_mean_absolute_error',
                    cv=KFold(5, shuffle=True, random_state=42),
                    n_jobs=-1, verbose=0
                )
                grid.fit(Xtr_s, y_train)
                best_params = grid.best_params_
                best_model  = grid.best_estimator_
            else:
                best_model = trained['Gradient Boosting']

            # Save to session state
            st.session_state.update({
                'result'      : result,
                'trained'     : trained,
                'Xtr_s'       : Xtr_s,
                'Xte_s'       : Xte_s,
                'y_train'     : y_train,
                'y_test'      : y_test,
                'scaler'      : scaler,
                'X_train'     : X_train,
                'X_test'      : X_test,
                'all_features': all_features,
                'brand_mean'  : brand_mean_tr,
                'global_mean' : global_mean,
                'best_model'  : best_model,
                'best_params' : best_params,
            })

        st.markdown("""
        <div class="box-success">
        ✅ <b>Training selesai!</b> Semua 6 model telah dilatih. Buka tab <b>Evaluasi</b> untuk analisis mendalam.
        </div>
        """, unsafe_allow_html=True)

    if 'result' in st.session_state:
        result    = st.session_state['result']
        best_name = result.iloc[0]['Model']

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="box-success">
        🏅 <b>Model Terbaik (MAE Test terendah):</b> &nbsp;<span class="model-badge">{best_name}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="sec-header">Tabel Perbandingan Model</p>', unsafe_allow_html=True)

        styled = result.round(4).style\
            .background_gradient(subset=['MAE Test', 'RMSE Test'], cmap='Reds')\
            .background_gradient(subset=['R² Test'], cmap='Greens')\
            .set_properties(**{'font-size': '13px'})
        st.dataframe(result.round(4), use_container_width=True, hide_index=True)

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.markdown('<p class="sec-header">Visualisasi Perbandingan Metrik</p>', unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ['MAE Test', 'RMSE Test', 'R² Test']
        palettes = ['#C0392B', '#E67E22', '#27AE60']
        for ax, met, pal in zip(axes, metrics, palettes):
            vals   = result.set_index('Model')[met]
            colors = [pal if m == best_name else '#BDC3C7' for m in vals.index]
            vals.plot(kind='bar', ax=ax, color=colors, edgecolor='white', width=0.65)
            ax.set_title(f'{met}', fontweight='bold')
            ax.set_ylabel(met)
            ax.tick_params(axis='x', rotation=35)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.4f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=7.5)
        plt.suptitle('Perbandingan 6 Model Regresi', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Evaluasi
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="sec-header">Evaluasi Model Terbaik — Gradient Boosting</p>', unsafe_allow_html=True)

    if 'best_model' not in st.session_state:
        st.markdown("""
        <div class="box-warning">
        ⚠️ Jalankan Training di tab <b>🤖 Training Model</b> terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
    else:
        best_model   = st.session_state['best_model']
        Xte_s        = st.session_state['Xte_s']
        Xtr_s        = st.session_state['Xtr_s']
        y_test       = st.session_state['y_test']
        y_train      = st.session_state['y_train']
        all_features = st.session_state['all_features']

        y_pred       = best_model.predict(Xte_s)
        y_pred_train = best_model.predict(Xtr_s)

        mae_test   = mean_absolute_error(y_test, y_pred)
        rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred))
        r2_test    = r2_score(y_test, y_pred)
        mae_train  = mean_absolute_error(y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        r2_train   = r2_score(y_train, y_pred_train)
        baseline   = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))

        if st.session_state.get('best_params'):
            st.markdown(f"""
            <div class="box-info">
            🔧 <b>Best Params (GridSearchCV):</b> <code>{st.session_state['best_params']}</code>
            </div>
            """, unsafe_allow_html=True)

        # KPI Metrics
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">📉</div>
                <div class="kpi-label">MAE Test</div>
                <div class="kpi-value">{mae_test:.4f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📊</div>
                <div class="kpi-label">RMSE Test</div>
                <div class="kpi-value">{rmse_test:.4f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-label">R² Test</div>
                <div class="kpi-value">{r2_test:.4f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📌</div>
                <div class="kpi-label">Baseline MAE</div>
                <div class="kpi-value">{baseline:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_tr1, col_tr2, col_tr3 = st.columns(3)
        col_tr1.metric("MAE Train",  f"{mae_train:.4f}",  delta=f"{mae_test-mae_train:+.4f} gap")
        col_tr2.metric("RMSE Train", f"{rmse_train:.4f}", delta=f"{rmse_test-rmse_train:+.4f} gap")
        col_tr3.metric("R² Train",   f"{r2_train:.4f}",   delta=f"{r2_test-r2_train:+.4f} gap")

        # Overfitting Analysis
        gap = r2_train - r2_test
        if gap > 0.3:
            box_cls, msg = "box-error",   "⚠️ Indikasi OVERFITTING kuat — gap R² > 0.3"
        elif gap > 0.1:
            box_cls, msg = "box-warning", "🟡 Overfitting RINGAN — wajar untuk ensemble pada dataset kecil"
        elif r2_test < 0:
            box_cls, msg = "box-warning", "❌ Indikasi UNDERFITTING — model lebih buruk dari baseline"
        else:
            box_cls, msg = "box-success", "✅ Model cukup seimbang antara train dan test"

        better_msg = "✅ Model LEBIH BAIK dari baseline." if mae_test < baseline else "⚠️ Model tidak lebih baik dari baseline."
        st.markdown(f'<div class="{box_cls}">{msg}<br>{better_msg}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

        # Aktual vs Prediksi
        st.markdown('<p class="sec-header">Aktual vs Prediksi & Distribusi Residual</p>', unsafe_allow_html=True)

        col_p, col_r = st.columns(2)
        with col_p:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, y_pred, alpha=0.35, color='#C0392B', s=20, label='Data Poin')
            lims = [min(y_test.min(), y_pred.min()) - 0.1, max(y_test.max(), y_pred.max()) + 0.1]
            ax.plot(lims, lims, 'k--', linewidth=1.5, label='Prediksi Sempurna')
            ax.set_xlabel('Rating Aktual')
            ax.set_ylabel('Rating Prediksi')
            ax.set_title('Aktual vs Prediksi (Test Set)')
            ax.legend(framealpha=0.8)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        with col_r:
            residuals = y_test.values - y_pred
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.hist(residuals, bins=25, color='#E74C3C', edgecolor='white', alpha=0.85)
            ax.axvline(0, color='#2C3E50', linestyle='--', lw=2)
            ax.axvline(residuals.mean(), color='#F39C12', linestyle=':', lw=2, label=f'Mean={residuals.mean():.3f}')
            ax.set_xlabel('Residual (Aktual – Prediksi)')
            ax.set_ylabel('Frekuensi')
            ax.set_title(f'Distribusi Residual (std={residuals.std():.3f})')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

        # Feature Importance
        if hasattr(best_model, 'feature_importances_'):
            st.markdown('<p class="sec-header">Feature Importance — Gradient Boosting</p>', unsafe_allow_html=True)

            imp = pd.Series(best_model.feature_importances_, index=all_features).sort_values()
            col_fi_a, col_fi_b = st.columns([3, 1])

            with col_fi_a:
                fig, ax = plt.subplots(figsize=(9, 5))
                colors_fi = plt.cm.Greens(np.linspace(0.3, 0.9, len(imp)))
                imp.plot(kind='barh', color=colors_fi, ax=ax, edgecolor='white')
                ax.bar_label(ax.containers[0], fmt='%.4f', fontsize=9, padding=4)
                ax.set_title('Feature Importance — Model Terbaik')
                ax.set_xlabel('Importance Score')
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            with col_fi_b:
                st.markdown("**📋 Tabel Importance**")
                st.dataframe(
                    imp.sort_values(ascending=False).rename("Importance")
                    .reset_index().rename(columns={'index': 'Fitur'}).round(4),
                    use_container_width=True, hide_index=True
                )

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.markdown('<p class="sec-header">Contoh 10 Prediksi pada Data Test</p>', unsafe_allow_html=True)

        result_df = st.session_state['X_test'].copy()
        result_df['rating_aktual']   = y_test.values
        result_df['rating_prediksi'] = y_pred.round(2)
        result_df['error']           = (result_df['rating_aktual'] - result_df['rating_prediksi']).round(3)
        result_df['abs_error']       = result_df['error'].abs().round(3)
        st.dataframe(
            result_df[['brand_avg_rating', 'harga_jual_avg', 'jumlah_size',
                        'rating_aktual', 'rating_prediksi', 'error', 'abs_error']].head(10),
            use_container_width=True
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Prediksi Manual
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="sec-header">Prediksi Rating Produk Baru</p>', unsafe_allow_html=True)

    if 'best_model' not in st.session_state:
        st.markdown("""
        <div class="box-warning">
        ⚠️ Jalankan Training di tab <b>🤖 Training Model</b> terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="box-info">
        💡 Masukkan detail produk skincare di bawah ini untuk memprediksi rating-nya menggunakan model Gradient Boosting yang telah dilatih.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("**💰 Informasi Harga**")
            harga_jual  = st.number_input("Harga Jual (Rp)", min_value=0, value=80_000, step=5_000, format="%d")
            harga_asli  = st.number_input("Harga Asli (Rp)", min_value=0, value=100_000, step=5_000, format="%d")
            pct_diskon  = st.number_input("Persentase Diskon (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
            jumlah_size = st.number_input("Jumlah Varian Ukuran", min_value=1, max_value=20, value=1)

        with col2:
            st.markdown("**🏷️ Informasi Produk**")
            nama_produk       = st.text_input("Nama Produk", value="Moisturizer Hyaluronic Acid 50ml")
            brand_input       = st.text_input("Brand", value="COSRX")
            brand_count_input = st.number_input("Jumlah Produk Brand Ini (di dataset)", min_value=1, value=20)

        st.markdown("<br>", unsafe_allow_html=True)
        pred_btn = st.button("🔮 Prediksi Rating Sekarang", type="primary", use_container_width=True)

        if pred_btn:
            scaler      = st.session_state['scaler']
            best_model  = st.session_state['best_model']
            brand_mean  = st.session_state['brand_mean']
            global_mean = st.session_state['global_mean']

            selisih   = harga_asli - harga_jual
            rasio     = harga_jual / harga_asli if harga_asli > 0 else 1.0
            nama_len  = len(nama_produk)
            brand_avg = float(brand_mean.get(brand_input, global_mean))

            inp = np.array([[
                harga_jual, harga_asli, pct_diskon,
                selisih, rasio, jumlah_size,
                nama_len, brand_count_input, brand_avg
            ]])
            inp_scaled = scaler.transform(inp)
            pred       = float(np.clip(best_model.predict(inp_scaled)[0], 2.6, 5.0))
            star_str   = stars(pred)
            brand_note = "(brand dikenal)" if brand_input in brand_mean.index else "(brand baru — menggunakan rata-rata global)"

            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-label">✨ Prediksi Rating untuk <b>{nama_produk}</b></div>
                <div class="pred-value">⭐ {pred:.2f}</div>
                <div class="pred-stars">{star_str}</div>
                <div class="pred-brand">Brand: <b>{brand_input}</b> {brand_note}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Detail Fitur yang Digunakan"):
                detail_df = pd.DataFrame({
                    'Fitur'      : FEATURE_COLS + ['brand_avg_rating'],
                    'Nilai Input': [
                        harga_jual, harga_asli, pct_diskon, selisih, rasio,
                        jumlah_size, nama_len, brand_count_input, brand_avg
                    ]
                })
                st.dataframe(detail_df.round(4), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="
    text-align: center;
    padding: 20px;
    border-top: 1px solid #eee;
    color: #bbb;
    font-size: 0.82rem;
    letter-spacing: 0.3px;
">
    🧴 <b>Sociola ML App</b> &nbsp;|&nbsp; Tugas Besar Machine Learning &nbsp;|&nbsp;
    Supervised Learning — Regresi Rating Produk Skincare &nbsp;|&nbsp; v2.0
</div>
""", unsafe_allow_html=True)
