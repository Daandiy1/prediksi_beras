import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime, timedelta
import random
import tensorflow as tf
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed untuk reproduksibilitas
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Konfigurasi halaman
st.set_page_config(
    page_title="🌾 Rice Price Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan yang menarik
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .step-header {
        background: linear-gradient(45deg, #2196F3, #1976D2);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        color: white;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
        color: black;
    }
    
    .success-box {
        background: #f0fff0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        color: black;
    }
    
    .warning-box {
        background: #fff8dc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        color: black;
    }
    
    .process-box {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #9c27b0;
        margin: 1rem 0;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown("""
<div class="main-header">
    <h1>🌾 Rice Price Forecasting System</h1>
    <p>Prediksi Harga Beras menggunakan Model Ensemble GRU-SVR dengan Data Cuaca</p>
</div>
""", unsafe_allow_html=True)

# Sidebar untuk navigasi
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Pilih Langkah:",
    [
        "🏠 Home", 
        "📂 Upload Data", 
        "🔧 Data Preprocessing", 
        "📊 Normalisasi Data",
        "✂️ Splitting Data",
        "🤖 Model Ensemble GRU-SVR",
        "📈 Evaluasi Model",
        "🔮 Prediksi"
    ]
)

# Fungsi utility
@st.cache_data
def load_data(uploaded_file):
    """Load data dari file CSV"""
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8',na_values=['-','0'])
        return df, True, "Data berhasil dimuat!"
    except Exception as e:
        return None, False, f"Error: {str(e)}"

def preprocess_data(df):
    """Preprocessing data"""
    try:
        df_processed = df.copy()
        
        # Konversi tanggal
        df_processed['Date'] = pd.to_datetime(df_processed['Date'], format='mixed', dayfirst=False)
        df_processed = df_processed.sort_values('Date')
        
        features_to_process = ['Harga_Beras_Medium', 'Tavg(°C)', 'RH_avg(%)', 'RR(mm)', 'ss(jam)']
        df_processed = df_processed[['Date'] + features_to_process].copy()
        
        # Preprocessing nilai anomali
        for col in features_to_process:
            df_processed[col] = df_processed[col].astype(str).str.replace(',', '.')
            df_processed[col] = df_processed[col].replace(['8888', '8888.0', '-', 'nan','0'], np.nan)
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Interpolasi untuk missing values
        df_processed.interpolate(method='linear', inplace=True)
        df_processed.fillna(method='bfill', inplace=True)
        df_processed.fillna(method='ffill', inplace=True)
        
        return df_processed, True, "Data berhasil dipreprocessing!"
    except Exception as e:
        return None, False, f"Error: {str(e)}"

def normalize_data(df_processed):
    """Normalisasi data menggunakan MinMaxScaler"""
    features_to_process = ['Harga_Beras_Medium', 'Tavg(°C)', 'RH_avg(%)', 'RR(mm)', 'ss(jam)']
    scalers = {}
    scaled_data = {}
    
    for col in features_to_process:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_processed[[col]])
        scalers[col] = scaler
        scaled_data[col] = scaled
    
    return scalers, scaled_data

def create_multivariate_sequences(target, features_data, window_size=7):
    """Membuat sequence untuk multivariate time series"""
    X, y = [], []
    for i in range(len(target) - window_size):
        window = np.hstack([features_data[col][i:i+window_size] for col in features_data])
        window = window.reshape(window_size, len(features_data))
        X.append(window)
        y.append(target[i+window_size])
    return np.array(X), np.array(y)

def build_gru_model(input_shape):
    """Membangun model GRU"""
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(GRU(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_feature_extractor(model):
    """Membuat feature extractor dari model GRU"""
    feature_extractor = Sequential()
    for i in range(len(model.layers) - 1):
        feature_extractor.add(model.layers[i])
    return feature_extractor

def evaluate_model(y_true, y_pred, scaler):
    """Evaluasi model"""
    actual = scaler.inverse_transform(y_true.reshape(-1, 1))
    predicted = scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    #r2 = r2_score(actual, predicted)
    
    return mape, mae, rmse, actual, predicted

def forecast_future(last_window_data, gru_feature_extractor, svr_model, price_scaler, 
                   avg_values, n_steps=7, n_features=5, window_size=7):
    """Forecasting untuk n_steps ke depan"""
    forecast_predictions_scaled = []
    current_sequence = last_window_data.copy()
    
    for _ in range(n_steps):
        sequence_for_gru = current_sequence.reshape(1, window_size, n_features)
        extracted_features = gru_feature_extractor.predict(sequence_for_gru, verbose=0)
        predicted_scaled_price = svr_model.predict(extracted_features)[0]
        forecast_predictions_scaled.append(predicted_scaled_price)
        
        new_row = current_sequence[-1, :].copy()
        new_row[0] = predicted_scaled_price
        new_row[1:] = avg_values
        
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, :] = new_row
    
    forecast_actual_prices = price_scaler.inverse_transform(
        np.array(forecast_predictions_scaled).reshape(-1, 1)
    )
    return forecast_actual_prices.flatten()

# ==================== HALAMAN ====================

# 1. HALAMAN HOME
if page == "🏠 Home":
    st.markdown("""
    <div class="step-header">
        <h2>🏠 Selamat Datang di Aplikasi Rice Price Forecasting System</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Tentang Aplikasi</h3>
        <p>Aplikasi ini menggunakan model <strong>Ensemble GRU-SVR</strong> untuk memprediksi harga beras 
        berdasarkan data historis harga dan kondisi cuaca. Model ini menggabungkan kekuatan 
        <strong>GRU (Gated Recurrent Unit)</strong> untuk ekstraksi fitur time series dan 
        <strong>SVR (Support Vector Regression)</strong> untuk prediksi akhir.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔧 Alur Sistem:
        1. **📂 Upload Data** - Upload dataset CSV harga beras
        2. **🔧 Data Preprocessing** - Pembersihan dan pengolahan data
        3. **📊 Normalisasi Data** - Scaling fitur ke rentang [0,1]
        4. **✂️ Splitting Data** - Pembagian data training dan testing
        5. **🤖 Model Ensemble** - Training model GRU dan SVR
        6. **📈 Evaluasi Model** - Pengujian performa model
        7. **🔮 Prediksi** - Forecasting 30 hari atau 7 Hari ke depan
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Format Data yang Diperlukan:
        - **Date** - Tanggal (format: YYYY-MM-DD)
        - **Harga_Beras_Medium** - Harga beras medium (Rp)
        - **Tavg(°C)** - Suhu rata-rata harian
        - **RH_avg(%)** - Kelembaban relatif rata-rata
        - **RR(mm)** - Curah hujan harian
        - **ss(jam)** - Lama penyinaran matahari
        """)
    
    st.markdown("""
    <div class="process-box">
        <h4>📝 Panduan Penggunaan:</h4>
        <ol>
            <li >Mulai dari halaman <strong>"Upload Data"</strong> untuk mengunggah file CSV</li>
            <li>Lanjutkan ke <strong>"Data Preprocessing"</strong> untuk melihat hasil pembersihan data</li>
            <li>Ikuti proses normalisasi dan splitting data</li>
            <li>Training model ensemble di halaman <strong>"Model Ensemble GRU-SVR"</strong></li>
            <li>Evaluasi performa model dan lihat hasil prediksi</li>
            <li>Dapatkan prediksi 30 hari atau 7 Hari ke depan di halaman <strong>"Prediksi"</strong></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# 2. HALAMAN UPLOAD DATA
elif page == "📂 Upload Data":
    st.markdown("""
    <div class="step-header">
        <h2>📂 Step 1: Upload Dataset</h2>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload file CSV dataset harga beras:",
        type=['csv'],
        help="File CSV harus berisi kolom: Date, Harga_Beras_Medium, Tavg(°C), RH_avg(%), RR(mm), ss(jam)"
    )
    
    if uploaded_file is not None:
        df_raw, success, message = load_data(uploaded_file)
        
        if success:
            st.session_state['df_raw'] = df_raw
            st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)
            
            # Informasi dataset
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Records", len(df_raw))
            with col2:
                st.metric("📋 Kolom", len(df_raw.columns))
            with col3:
                st.metric("💾 Ukuran File", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Preview data mentah
            st.markdown("### 👀 Preview Data Mentah")
            st.dataframe(df_raw.head(10), use_container_width=True)
            
            # Info struktur data
            st.markdown("### 📋 Informasi Struktur Data")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Nama Kolom:**")
                for i, col in enumerate(df_raw.columns, 1):
                    st.write(f"{i}. {col}")
            
            with col2:
                st.markdown("**Tipe Data:**")
                data_types = df_raw.dtypes
                for col in df_raw.columns:
                    st.write(f"• {col}: {data_types[col]}")
            
            # Missing values check
            missing_data = df_raw.isnull().sum()
            if missing_data.sum() > 0:
                st.markdown("### ⚠️ Missing Values Detected")
                st.dataframe(missing_data[missing_data > 0], use_container_width=True)
            else:
                st.success("✅ Tidak ada missing values terdeteksi!")
                
        else:
            st.markdown(f'<div class="warning-box">❌ {message}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box",>
            <p>📋 Silakan upload file CSV dataset untuk memulai proses analisis.</p>
            <p><strong>Format yang dibutuhkan:</strong> CSV dengan kolom Date, Harga_Beras_Medium, Tavg(°C), RH_avg(%), RR(mm), ss(jam)</p>
        </div>
        """, unsafe_allow_html=True)

# 3. HALAMAN DATA PREPROCESSING
elif page == "🔧 Data Preprocessing":
    st.markdown("""
    <div class="step-header">
        <h2>🔧 Step 2: Data Preprocessing</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if 'df_raw' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Silakan upload dataset terlebih dahulu di halaman "Upload Data".
        </div>
        """, unsafe_allow_html=True)
    else:
        df_raw = st.session_state['df_raw']
        
        st.markdown("""
        <div class="process-box">
            <h4>🔄 Proses Preprocessing meliputi:</h4>
            <ul>
                <li>Konversi format tanggal</li>
                <li>Sorting data berdasarkan tanggal</li>
                <li>Pembersihan nilai anomali (0,8888, -, nan)</li>
                <li>Konversi tipe data</li>
                <li>Interpolasi linear untuk missing values</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔄 Melakukan preprocessing data..."):
            df_processed, success, message = preprocess_data(df_raw)
        
        if success:
            st.session_state['df_processed'] = df_processed
            st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)
            
            # Perbandingan sebelum dan sesudah preprocessing
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Data Sebelum Preprocessing")
                st.metric("Missing Values", df_raw.isnull().sum().sum())
                st.dataframe(df_raw.head(), use_container_width=True)
            
            with col2:
                st.markdown("### ✨ Data Setelah Preprocessing")
                st.metric("Missing Values", df_processed.isnull().sum().sum())
                st.dataframe(df_processed.head(), use_container_width=True)
            
            # Informasi periode data
            st.markdown("### 📅 Informasi Periode Data")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🗓️ Tanggal Mulai", df_processed['Date'].min().strftime('%Y-%m-%d'))
            with col2:
                st.metric("🗓️ Tanggal Akhir", df_processed['Date'].max().strftime('%Y-%m-%d'))
            with col3:
                date_range = df_processed['Date'].max() - df_processed['Date'].min()
                st.metric("📊 Total Hari", f"{date_range.days} hari")
            
            # Statistik deskriptif
            st.markdown("### 📈 Statistik Deskriptif Data Clean")
            st.dataframe(df_processed.describe().round(3), use_container_width=True)
            
            # Visualisasi data clean
            st.markdown("### 📊 Visualisasi Data Setelah Preprocessing")
            features = ['Harga_Beras_Medium', 'Tavg(°C)', 'RH_avg(%)', 'RR(mm)', 'ss(jam)']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, col in enumerate(features):
                axes[i].plot(df_processed['Date'], df_processed[col], linewidth=1, alpha=0.8)
                axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Tanggal')
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='x', rotation=45)
            
            axes[5].axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.markdown(f'<div class="warning-box">❌ {message}</div>', unsafe_allow_html=True)

# 4. HALAMAN NORMALISASI DATA
elif page == "📊 Normalisasi Data":
    st.markdown("""
    <div class="step-header">
        <h2>📊 Step 3: Normalisasi Data</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if 'df_processed' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Silakan lakukan preprocessing data terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
    else:
        df_processed = st.session_state['df_processed']
        
        st.markdown("""
        <div class="process-box">
            <h4>🎯 Tujuan Normalisasi:</h4>
            <ul>
                <li>Mengubah skala semua fitur ke rentang [0, 1]</li>
                <li>Mencegah dominasi fitur dengan nilai besar</li>
                <li>Meningkatkan performa algoritma machine learning</li>
                <li>Mempercepat konvergensi model</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("📊 Melakukan normalisasi data..."):
            scalers, scaled_data = normalize_data(df_processed)
        
        st.session_state['scalers'] = scalers
        st.session_state['scaled_data'] = scaled_data
        
        st.success("✅ Normalisasi data selesai!")
        
        # Perbandingan data sebelum dan sesudah normalisasi
        features = ['Harga_Beras_Medium', 'Tavg(°C)', 'RH_avg(%)', 'RR(mm)', 'ss(jam)']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Statistik Data Asli")
            original_stats = df_processed[features].describe().round(3)
            st.dataframe(original_stats, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Statistik Data Normalisasi")
            normalized_df = pd.DataFrame({col: scaled_data[col].flatten() for col in features})
            normalized_stats = normalized_df.describe().round(3)
            st.dataframe(normalized_stats, use_container_width=True)
        
        # Visualisasi perbandingan
        st.markdown("### 📊 Perbandingan Data Asli vs Normalisasi")
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        
        for i, col in enumerate(features):
            # Data asli
            axes[0, i].plot(df_processed['Date'], df_processed[col], alpha=0.8, color='blue')
            axes[0, i].set_title(f'{col} (Original)', fontweight='bold')
            axes[0, i].set_xlabel('Tanggal')
            axes[0, i].tick_params(axis='x', rotation=45)
            axes[0, i].grid(True, alpha=0.3)
            
            # Data normalisasi
            axes[1, i].plot(df_processed['Date'], scaled_data[col], alpha=0.8, color='red')
            axes[1, i].set_title(f'{col} (Normalized)', fontweight='bold')
            axes[1, i].set_xlabel('Tanggal')
            axes[1, i].set_ylabel('Normalized Value [0,1]')
            axes[1, i].tick_params(axis='x', rotation=45)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# 5. HALAMAN SPLITTING DATA
elif page == "✂️ Splitting Data":
    st.markdown("""
    <div class="step-header">
        <h2>✂️ Step 4: Splitting Data</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if 'scaled_data' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Silakan lakukan normalisasi data terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
    else:
        df_processed = st.session_state['df_processed']
        scaled_data = st.session_state['scaled_data']
        
        st.markdown("""
        <div class="process-box">
            <h4>🎯 Konfigurasi Splitting:</h4>
            <ul>
                <li><strong>Window Size:</strong> 7 hari (untuk sequence input)</li>
                <li><strong>Training Ratio:</strong> 80% data untuk training</li>
                <li><strong>Testing Ratio:</strong> 20% data untuk testing</li>
                <li><strong>Metode:</strong> Time series split (chronological order)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("✂️ Melakukan splitting data..."):
            # Membuat sequences
            window_size = 7
            features_to_process = ['Harga_Beras_Medium', 'Tavg(°C)', 'RH_avg(%)', 'RR(mm)', 'ss(jam)']
            
            X_full, y_full = create_multivariate_sequences(
                scaled_data['Harga_Beras_Medium'],
                {col: scaled_data[col] for col in features_to_process},
                window_size
            )
            
            # Split data
            train_ratio = 0.8
            split_idx = int(train_ratio * len(X_full))
            
            X_train, X_test = X_full[:split_idx], X_full[split_idx:]
            y_train, y_test = y_full[:split_idx], y_full[split_idx:]
            
            # Simpan dates
            dates = df_processed['Date'][window_size:].reset_index(drop=True)
            dates_train = dates[:split_idx]
            dates_test = dates[split_idx:]
        
        # Simpan ke session state
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['dates_train'] = dates_train
        st.session_state['dates_test'] = dates_test
        st.session_state['window_size'] = window_size
        
        st.success("✅ Data splitting selesai!")
        
        # Informasi hasil splitting
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Total Sequences", len(X_full))
        with col2:
            st.metric("📚 Training Data", len(X_train))
        with col3:
            st.metric("🧪 Testing Data", len(X_test))
        with col4:
            st.metric("📏 Window Size", window_size)
        
        # Visualisasi pembagian data
        st.markdown("### 📊 Visualisasi Pembagian Data")
        
        # Ambil harga asli untuk visualisasi
        scalers = st.session_state['scalers']
        price_scaler = scalers['Harga_Beras_Medium']
        
        y_train_actual = price_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_actual = price_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        ax.plot(dates_train, y_train_actual, label='Training Data', color='blue', alpha=0.8)
        ax.plot(dates_test, y_test_actual, label='Testing Data', color='red', alpha=0.8)
        ax.axvline(x=dates_test.iloc[0], color='black', linestyle='--', alpha=0.7, label='Split Point')
        
        ax.set_title('Pembagian Data Training dan Testing', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Harga Beras (Rp)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detail periode
        st.markdown("### 📅 Detail Periode Data")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟦 Training Period**")
            st.write(f"📅 Mulai: {dates_train.min().strftime('%Y-%m-%d')}")
            st.write(f"📅 Akhir: {dates_train.max().strftime('%Y-%m-%d')}")
            st.write(f"📊 Total: {len(dates_train)} hari")
        
        with col2:
            st.markdown("**🟥 Testing Period**")
            st.write(f"📅 Mulai: {dates_test.min().strftime('%Y-%m-%d')}")
            st.write(f"📅 Akhir: {dates_test.max().strftime('%Y-%m-%d')}")
            st.write(f"📊 Total: {len(dates_test)} hari")

# 6. HALAMAN MODEL ENSEMBLE
elif page == "🤖 Model Ensemble GRU-SVR":
    st.markdown("""
    <div class="step-header">
        <h2>🤖 Step 5: Training Model Ensemble GRU-SVR</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if 'X_train' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Silakan lakukan splitting data terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
    else:
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
        window_size = st.session_state['window_size']
        
        st.markdown("""
        <div class="process-box">
            <h4>🧠 Arsitektur Model Ensemble:</h4>
            <ol>
                <li><strong>GRU Model:</strong> Ekstraksi fitur temporal dari sequence data</li>
                <li><strong>Feature Extraction:</strong> Mengambil output layer sebelum dense</li>
                <li><strong>SVR Model:</strong> Regresi menggunakan fitur yang diekstrak GRU</li>
                <li><strong>Hyperparameter Tuning:</strong> Grid search untuk parameter SVR optimal</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔧 Parameter GRU:**")
            st.write("• Hidden Units: 64")
            st.write("• Dropout Rate: 0.2")
            st.write("• Optimizer: Adam")
            st.write("• Loss Function: MSE")
            
        with col2:
            st.markdown("**🔧 Parameter SVR Grid Search:**")
            st.write("• Kernel: RBF")
            st.write("• C: [10, 100, 500, 1000]")
            st.write("• Gamma: [0.001, 0.01, 0.1, 1]")
            st.write("• Total Kombinasi: 16")
        
        if st.button("🚀 Mulai Training Model Ensemble", type="primary"):
            
            # Training GRU
            st.markdown("### 🧠 Step 5.1: Training Model GRU")
            with st.spinner("🔄 Training GRU model..."):
                
                y_train_flat = y_train.ravel()
                y_test_flat = y_test.ravel()
                
                # Build GRU model
                gru_model = build_gru_model((window_size, 5))
                
                # Callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
                ]
                
                # Training
                history = gru_model.fit(
                    X_train, y_train_flat,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test_flat),
                    callbacks=callbacks,
                    verbose=0
                )
            
            st.success("✅ GRU training selesai!")
            
            # Visualisasi training history
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history.history['loss'], label='Train Loss', color='blue')
            ax.plot(history.history['val_loss'], label='Validation Loss', color='red')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.set_title('GRU Training History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Feature extraction
            st.markdown("### 🔍 Step 5.2: Feature Extraction")
            with st.spinner("🔄 Extracting features..."):
                feature_extractor = create_feature_extractor(gru_model)
                features_train = feature_extractor.predict(X_train, verbose=0)
                features_test = feature_extractor.predict(X_test, verbose=0)
            
            st.success(f"✅ Feature extraction selesai! Shape: {features_train.shape}")
            
            # SVR Grid Search
            st.markdown("### 🎯 Step 5.3: SVR Hyperparameter Tuning")
            
            C_values = [10, 100, 500, 1000]
            gamma_values = [0.001, 0.01, 0.1, 1]
            
            best_mape = float('inf')
            best_model = None
            best_params = None
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_combinations = len(C_values) * len(gamma_values)
            current_combination = 0
            
            scalers = st.session_state['scalers']
            
            results_data = []
            
            for C in C_values:
                for gamma in gamma_values:
                    current_combination += 1
                    status_text.text(f"Testing C={C}, gamma={gamma} ({current_combination}/{total_combinations})")
                    
                    svr = SVR(kernel='rbf', C=C, gamma=gamma)
                    svr.fit(features_train, y_train_flat)
                    pred = svr.predict(features_test)
                    
                    mape, mae, rmse, _, _ = evaluate_model(
                        y_test, pred.reshape(-1, 1), scalers['Harga_Beras_Medium']
                    )
                    
                    results_data.append({
                        'C': C,
                        'gamma': gamma,
                        'MAPE': mape,
                        'MAE': mae,
                        'RMSE': rmse,
                        
                    })
                    
                    if mape < best_mape:
                        best_mape = mape
                        best_model = svr
                        best_params = (C, gamma)
                    
                    progress_bar.progress(current_combination / total_combinations)
            
            status_text.text("✅ Grid search selesai!")
            
            # Simpan model terbaik
            st.session_state['best_gru_model'] = gru_model
            st.session_state['best_svr_model'] = best_model
            st.session_state['feature_extractor'] = feature_extractor
            st.session_state['training_history'] = history
            
            st.success("🎉 Model Ensemble berhasil ditraining!")
            
            # Hasil grid search
            st.markdown("### 📊 Hasil Hyperparameter Tuning")
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df.round(4), use_container_width=True)
            
            # Best parameters
            st.markdown(f"""
            <div class="success-box">
                <h4>🏆 Parameter Terbaik:</h4>
                <p><strong>C:</strong> {best_params[0]} | <strong>Gamma:</strong> {best_params[1]} | <strong>MAPE:</strong> {best_mape:.3f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Heatmap hasil grid search
            st.markdown("### 🔥 Heatmap Hasil Grid Search (MAPE)")
            pivot_table = results_df.pivot(index='gamma', columns='C', values='MAPE')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_table, annot=True, cmap='RdYlBu_r', fmt='.3f', ax=ax)
            ax.set_title('Grid Search Results - MAPE Values')
            st.pyplot(fig)

# 7. HALAMAN EVALUASI MODEL
elif page == "📈 Evaluasi Model":
    st.markdown("""
    <div class="step-header">
        <h2>📈 Step 6: Evaluasi Model dan Visualisasi</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if 'best_svr_model' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Silakan training model ensemble terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Ambil model dan data
        best_svr_model = st.session_state['best_svr_model']
        feature_extractor = st.session_state['feature_extractor']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        dates_test = st.session_state['dates_test']
        scalers = st.session_state['scalers']
        
        st.markdown("""
        <div class="process-box">
            <h4>📊 Metrik Evaluasi:</h4>
            <ul>
                <li><strong>MAPE (Mean Absolute Percentage Error):</strong> Kesalahan persentase rata-rata</li>
                <li><strong>MAE (Mean Absolute Error):</strong> Kesalahan absolut rata-rata</li>
                <li><strong>RMSE (Root Mean Square Error):</strong> Akar kuadrat kesalahan rata-rata</li>   
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("📊 Mengevaluasi model..."):
            # Prediksi
            features_test = feature_extractor.predict(X_test, verbose=0)
            y_pred = best_svr_model.predict(features_test)
            
            # Evaluasi
            mape, mae, rmse, actual, predicted = evaluate_model(
                y_test, y_pred.reshape(-1, 1), scalers['Harga_Beras_Medium']
            )
        
        # Tampilkan metrik
        st.markdown("### 🎯 Hasil Evaluasi Model")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 MAPE", f"{mape:.3f}%")
        with col2:
            st.metric("📊 MAE", f"{mae:.3f}")
        with col3:
            st.metric("📊 RMSE", f"{rmse:.3f}")
        #with col4:
            #st.metric("🎯 R²", f"{r2:.3f}")
        
        # Interpretasi hasil
        if mape < 10:
            interpretation = "🟢 Excellent - Model memiliki akurasi sangat baik"
        elif mape < 20:
            interpretation = "🟡 Good - Model memiliki akurasi yang baik"
        else:
            interpretation = "🔴 Fair - Model perlu perbaikan"
        
        st.markdown(f"""
        <div class="info-box">
            <h4>💡 Interpretasi Hasil:</h4>
            <p>{interpretation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi prediksi vs aktual
        st.markdown("### 📊 Visualisasi Prediksi vs Data Aktual")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Time series comparison
        ax1.plot(dates_test, actual.flatten(), label='Data Aktual', marker='o', markersize=4, alpha=0.8, color='blue')
        ax1.plot(dates_test, predicted.flatten(), label='Prediksi Model', marker='x', markersize=4, alpha=0.8, color='red')
        ax1.set_title(f'Perbandingan Prediksi vs Aktual (MAPE: {mape:.3f}%', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Tanggal')
        ax1.set_ylabel('Harga Beras (Rp)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Scatter plot
        ax2.scatter(actual.flatten(), predicted.flatten(), alpha=0.6, color='green')
        ax2.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label='Perfect Prediction')
        ax2.set_xlabel('Harga Aktual (Rp)')
        ax2.set_ylabel('Harga Prediksi (Rp)')
        ax2.set_title('Scatter Plot: Prediksi vs Aktual')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Error analysis
        st.markdown("### 📊 Analisis Error")
        
        errors = predicted.flatten() - actual.flatten()
        percentage_errors = (errors / actual.flatten()) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(percentage_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Percentage Error (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribusi Percentage Error')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(dates_test, percentage_errors, marker='o', markersize=3, alpha=0.7, color='orange')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Percentage Error (%)')
            ax.set_title('Percentage Error Over Time')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        
        # Detail error statistics
        st.markdown("### 📈 Statistik Error")
        error_stats = pd.DataFrame({
            'Metrik': ['Mean Error', 'Std Error', 'Min Error', 'Max Error', 'Mean Abs Error %'],
            'Nilai': [f"{errors.mean():.2f}", f"{errors.std():.2f}", f"{errors.min():.2f}", 
                     f"{errors.max():.2f}", f"{abs(percentage_errors).mean():.2f}%"]
        })
        st.dataframe(error_stats, use_container_width=True)

# 8. HALAMAN PREDIKSI
elif page == "🔮 Prediksi":
    st.markdown("""
    <div class="step-header">
        <h2>🔮 Step 7: Prediksi Harga 30 Hari ke Depan atau 7 Hari Kedepan</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if 'best_svr_model' not in st.session_state:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Silakan training dan evaluasi model terlebih dahulu.
        </div>
        """, unsafe_allow_html=True)
    else:
        df_processed = st.session_state['df_processed']
        scalers = st.session_state['scalers']
        scaled_data = st.session_state['scaled_data']
        
        st.markdown("""
        <div class="process-box">
            <h4>🎯 Proses Forecasting:</h4>
            <ol>
                <li>Training ulang model dengan seluruh data</li>
                <li>Menggunakan 7 hari terakhir sebagai input awal</li>
                <li>Prediksi iteratif untuk 30 hari ke depan atau 7 hari kedepan</li>
                <li>Asumsi cuaca menggunakan nilai rata-rata historis</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
    # Buat dua kolom untuk tombol
    col1, col2 = st.columns(2)

    # Inisialisasi variabel durasi prediksi
    days_to_predict = 0

    with col1:
        if st.button("📅 Mulai Prediksi 7 Hari", type="primary"):
            days_to_predict = 7

    with col2:
        if st.button("🚀 Mulai Prediksi 30 Hari", type="primary"):
            days_to_predict = 30

    # Blok utama akan berjalan jika salah satu tombol ditekan (days_to_predict > 0)
    if days_to_predict > 0:
    
        with st.spinner(f"🔮 Melakukan forecasting untuk {days_to_predict} hari..."):
        # Retrain dengan seluruh data (langkah ini sama untuk kedua durasi)
            features_to_process = ['Harga_Beras_Medium', 'Tavg(°C)', 'RH_avg(%)', 'RR(mm)', 'ss(jam)']
            window_size = 7
        
            X_full, y_full = create_multivariate_sequences(
                scaled_data['Harga_Beras_Medium'],
                {col: scaled_data[col] for col in features_to_process},
                window_size
            )
        
            y_full_flat = y_full.ravel()
            dates_full_target = df_processed['Date'][window_size:].reset_index(drop=True)
        
            # Train GRU final
            gru_model_final = build_gru_model((window_size, len(features_to_process)))
        
            callbacks = [
                EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
            ]
        
            gru_model_final.fit(
                X_full, y_full_flat,
                epochs=11,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
        
        # Feature extraction dan SVR final
        feature_extractor_final = create_feature_extractor(gru_model_final)
        features_extracted_full = feature_extractor_final.predict(X_full, verbose=0)
        
        best_svr_model = st.session_state['best_svr_model']
        svr_model_final = SVR(kernel='rbf', C=best_svr_model.C, gamma=best_svr_model.gamma)
        svr_model_final.fit(features_extracted_full, y_full_flat)
        
        # Hitung nilai rata-rata untuk fitur cuaca
        avg_temp_scaled = X_full[:, :, 1].mean()
        avg_rh_scaled = X_full[:, :, 2].mean()
        avg_rr_scaled = X_full[:, :, 3].mean()
        avg_ss_scaled = X_full[:, :, 4].mean()
        avg_values = [avg_temp_scaled, avg_rh_scaled, avg_rr_scaled, avg_ss_scaled]
        
        # Forecasting (Gunakan variabel days_to_predict)
        last_window_for_forecast = X_full[-1]
        forecasted_prices = forecast_future(
            last_window_for_forecast,
            feature_extractor_final,
            svr_model_final,
            scalers['Harga_Beras_Medium'],
            avg_values,
            n_steps=days_to_predict  # Variabel dinamis
        )
        
        last_known_date = df_processed['Date'].iloc[-1]
        forecast_dates = [last_known_date + timedelta(days=i+1) for i in range(days_to_predict)]
        
        # Prediksi historis untuk visualisasi
        historical_svr_preds_scaled_full = svr_model_final.predict(features_extracted_full)
        historical_actual_prices_full = scalers['Harga_Beras_Medium'].inverse_transform(
            y_full_flat.reshape(-1, 1)
        ).flatten()
        historical_svr_preds_actual_full = scalers['Harga_Beras_Medium'].inverse_transform(
            historical_svr_preds_scaled_full.reshape(-1, 1)
        ).flatten()
    
        st.success(f"✅ Prediksi {days_to_predict} hari selesai!")
    
        # Statistik prediksi (Gunakan variabel days_to_predict)
        st.markdown(f"### 📊 Statistik Prediksi {days_to_predict} Hari")
    
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Harga Tertinggi", f"Rp {forecasted_prices.max():,.3f}")
        with col2:
            st.metric("💰 Harga Terendah", f"Rp {forecasted_prices.min():,.3f}")
        with col3:
            st.metric("💰 Harga Rata-rata", f"Rp {forecasted_prices.mean():,.3f}")
        with col4:
            current_price = historical_actual_prices_full[-1]
            avg_forecast = forecasted_prices.mean()
            trend = "📈" if avg_forecast > current_price else "📉"
            change_pct = ((avg_forecast - current_price) / current_price) * 100
            st.metric("📈 Trend", f"{trend} {change_pct:+.2f}%")
    
        # Visualisasi lengkap (Gunakan variabel days_to_predict)
        st.markdown("### 📊 Visualisasi Lengkap (Historis + Prediksi)")

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(dates_full_target, historical_actual_prices_full, 
                label='Data Aktual Historis', 
                marker='o', markersize=3, alpha=0.8, color='blue')
        ax.plot(dates_full_target, historical_svr_preds_actual_full, 
                label='Prediksi Model Historis', 
                marker='x', markersize=2, linestyle='--', alpha=0.7, color='green')
        ax.plot(forecast_dates, forecasted_prices, 
                label=f'Prediksi {days_to_predict} Hari', marker='D', markersize=4, 
                linestyle='-', color='red', linewidth=2)
        ax.axvline(x=last_known_date, color='black', linestyle=':', alpha=0.7, label='Batas Prediksi')
        ax.set_title(f'Prediksi Harga Beras Medium - {days_to_predict} Hari ke Depan', fontsize=16, fontweight='bold')
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Harga (Rp)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
        # Visualisasi hanya prediksi (Gunakan variabel days_to_predict)
        st.markdown(f"### 🔮 Detail Prediksi {days_to_predict} Hari")
    
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(forecast_dates, forecasted_prices, 
                marker='D', markersize=6, linestyle='-', color='red', linewidth=2)
    
        uncertainty = forecasted_prices.std() * 0.1
        upper_bound = forecasted_prices + uncertainty
        lower_bound = forecasted_prices - uncertainty
        ax.fill_between(forecast_dates, lower_bound, upper_bound, alpha=0.2, color='red')
    
        ax.set_title(f'Prediksi Harga Beras Medium - {days_to_predict} Hari ke Depan', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tanggal', fontsize=12)
        ax.set_ylabel('Harga (Rp)', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
        # Tabel prediksi detail
        st.markdown("### 📋 Detail Prediksi Harga per Hari")
        # 1. Menghitung Tren Harian
        daily_trends = []
        # Ambil harga aktual terakhir sebagai pembanding untuk hari pertama
        last_actual_price = historical_actual_prices_full[-1] 

        for i, current_forecast_price in enumerate(forecasted_prices):
        #Tentukan harga pembanding (hari sebelumnya)
            if i == 0:
                previous_price = last_actual_price
            else:
                previous_price = forecasted_prices[i - 1]

            # Hitung perubahan dan persentase
            change = current_forecast_price - previous_price
            # Hindari ZeroDivisionError jika harga sebelumnya 0
            if previous_price != 0:
                pct_change = (change / previous_price) * 100
            else:
                pct_change = float('inf') # Atau 0, tergantung cara Anda ingin menampilkannya
    
            # Tentukan ikon tren
            if change > 0:
                trend_icon = "📈"  # Naik
            elif change < 0:
                trend_icon = "📉"  # Turun
            else:
                trend_icon = "➡️"  # Stabil

            daily_trends.append(f"{trend_icon} {pct_change:+.2f}%")

        # 2. Membuat DataFrame dengan kolom baru "Tren Harian"
        forecast_df = pd.DataFrame({
            'Tanggal': forecast_dates,
            'Hari': [d.strftime('%A') for d in forecast_dates],
            'Prediksi Harga (Rp)': [f"Rp {price:,.3f}" for price in forecasted_prices],
            'Tren Harian': daily_trends,  # Kolom baru ditambahkan di sini
            'Minggu': [f'Minggu {(i // 7) + 1}' for i in range(days_to_predict)]
        })
    
        st.dataframe(forecast_df, use_container_width=True, 
                    column_config={
                    "Prediksi Harga (Rp)": st.column_config.TextColumn(width="medium"),
                    "Tren Harian": st.column_config.TextColumn(width="small")
                })
    
    # Analisis tren per minggu (Dibuat lebih dinamis)
    num_weeks = (days_to_predict + 6) // 7 # Pembulatan ke atas untuk jumlah minggu
    if num_weeks > 1:
        st.markdown("### 📈 Analisis Tren per Minggu")
        
        weekly_analysis = []
        for week in range(1, num_weeks + 1):
            start_idx = (week - 1) * 7
            end_idx = min(week * 7, days_to_predict)
            week_prices = forecasted_prices[start_idx:end_idx]
            
            if len(week_prices) > 0: # Pastikan ada data di minggu tersebut
                weekly_analysis.append({
                    'Minggu': f'Minggu {week}',
                    'Rata-rata Harga': f"Rp {week_prices.mean():,.3f}",
                    'Harga Tertinggi': f"Rp {week_prices.max():,.3f}",
                    'Harga Terendah': f"Rp {week_prices.min():,.3f}",
                    'Volatilitas': f"Rp {week_prices.std():,.3f}" if len(week_prices) > 1 else "Rp 0.000"
                })
        
        weekly_df = pd.DataFrame(weekly_analysis)
        st.dataframe(weekly_df, use_container_width=True)
    
        # Download hasil (Gunakan variabel days_to_predict)
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download Hasil Prediksi ({days_to_predict} Hari) CSV",
            data=csv,
            file_name=f"forecast_harga_beras_{days_to_predict}hari_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🌾 Rice Price Forecasting System | Powered by Ensemble GRU-SVR Model</p>
    <p>Built with Streamlit & TensorFlow | Step-by-Step Machine Learning Pipeline</p>
</div>
""", unsafe_allow_html=True)