import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Konfigurasi halaman
st.set_page_config(page_title=" Aplikasi Klasifikasi Kualitas Beras", layout="centered")

st.markdown("""
    <style>
    .sidebar .sidebar-content {
        padding-top: 2rem;
    }
     section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        font-size: 22px !important;
        font-weight: 700 !important;
        padding: 6px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Label encoding
label_encode = {"Premium": 2, "Medium": 1, "Rendah": 0}
label_decode = {v: k for k, v in label_encode.items()}

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset coba (2).csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()
X = df[['warna', 'ukuran', 'keutuhan', 'kebersihan']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Navigasi
menu = st.sidebar.radio(
    "📁 Navigasi",
    ["🏠 Beranda", "📊 Klasifikasi"],
    label_visibility="collapsed"
)

if menu == "🏠 Beranda":
    st.markdown("""
    <div style='text-align:center;'>
        <h1 style='font-size:42px; font-weight:700;'>🌾Aplikasi Klasifikasi Kualitas Beras</h1>
        <p style='font-size:19px; text-align:center; margin-top:-10px;'>Membantu Anda menilai kualitas beras berdasarkan parameter fisik secara cepat dan akurat. Jika ingin langsung melakukan klasifikasi, maka tekan bagian "📊 Klasifikasi" yang terletak di sebelah kiri halaman.</p>
    </div>
    """, unsafe_allow_html=True)

    # Card deskripsi sistem
    st.markdown("""
    <div style='background-color:#f9f9f9; padding:20px; border-radius:12px; margin-top:20px; color: black;'>
        <h3>🛠️ Tentang Sistem</h3>
        <p style='font-size:21px; text-align:center;'>Aplikasi ini menggunakan algoritma Decision Tree untuk mengklasifikasikan beras menjadi kategori <b>Premium</b>, <b>Medium</b>, atau <b>Rendah</b>. 
        Input dilakukan berdasarkan parameter fisik beras seperti warna, ukuran, keutuhan, dan kebersihan. 
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Importance
    st.markdown("""
    <div style='margin-top:40px;'>
        <h3>📊 Pengaruh Parameter terhadap Kualitas Beras</h3>
    </div>
    """, unsafe_allow_html=True)

    fitur_nama = ['Warna', 'Ukuran', 'Keutuhan', 'Kebersihan']
    importance = [0.25, 0.25, 0.30, 0.20] 

    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.barplot(x=importance, y=fitur_nama, ax=ax, orient='h', palette="Greens_d")
    ax.set_xlim(0, 0.35)
    ax.set_title("Tingkat Pengaruh (%)", fontsize=13)
    st.pyplot(fig)

    # Penjelasan kategori
    st.markdown("<h3 style='margin-top:40px;'>📄 Kategori Kualitas Beras (SNI 6128:2020)</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background-color:#e8f5e9; padding:15px; border-radius:10px; color: black;'>
            <h4 style='color:#2e7d32;'>🌟Premium</h4>
            <ul style='font-size:20px;'>
                <li>Warna cerah</li>
                <li>Ukuran seragam</li>
                <li>Keutuhan Beras utuh</li>
                <li>Bersih</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color:#fff3e0; padding:15px; border-radius:10px; color: black;'>
            <h4 style='color:#ef6c00;'>〽️Medium</h4>
            <ul style='font-size:20px;'>
                <li>Warna cukup cerah</li>
                <li>Ukuran sedikit bervariasi</li>
                <li>Keutuhan Beras cukup utuh</li>
                <li>Cukup bersih</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background-color:#ffebee; padding:15px; border-radius:10px; color: black;'>
            <h4 style='color:#c62828;'>⚠️Rendah</h4>
            <ul style='font-size:20px;'>
                <li>Warna kusam</li>
                <li>Ukuran tidak seragam</li>
                <li>Keutuhan Beras tidak utuh</li>
                <li>Kotor</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif menu == "📊 Klasifikasi":
    st.markdown("""
    <div style='text-align:center;'>
        <h1 style='font-size:42px; font-weight:700;'>🌾Aplikasi Klasifikasi Kualitas Beras</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <h4 style='text-align: center; font-size: 23px;'>Selamat datang Pada Halaman Klasifikasi Kualitas Beras</h4> 
    <div style='background-color:#f9f9f9; padding:20px; border-radius:12px; margin-top:20px; color: black;'>
    <p style='text-align: center; font-size: 18px;'>
        <b>Aplikasi ini dirancang untuk membantu Anda menilai dan mengklasifikasikan kualitas beras berdasarkan 
        parameter fisik yang umum digunakan, yaitu: Warna, Ukuran, 
        Keutuhan, dan Kebersihan.</b>
    </p>
    <p style='text-align: center; font-size: 18px;'>
        <b>Sebelum Anda memberikan penilaian terhadap kualitas beras, pastikan Anda memahami terlebih dahulu karakteristik dari masing-masing parameter fisik.
        Untuk memudahkan proses penilaian, Disediakan contoh visual yang bisa Anda jadikan referensi dalam menilai beras berdasarkan warna, ukuran, keutuhan, dan kebersihannya.
        Silakan perhatikan gambar-gambar berikut sebagai referensi penilaian dengan skala dari 1 (terburuk) hingga 3 (terbaik).</b>
    </p>
""", unsafe_allow_html=True)


    st.subheader("Warna")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("warna1.jpg", caption="1 - Kusam", use_container_width=True)
    with col2:
        st.image("warna2.jpg", caption="2 - Cukup Cerah", use_container_width=True)
    with col3:
        st.image("warna3.jpg", caption="3 - Cerah", use_container_width=True)

    st.subheader("Ukuran")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("ukuran1.jpg", caption="1 - Tidak Seragam", use_container_width=True)
    with col2:
        st.image("ukuran2.jpg", caption="2 - Sedikit Variasi", use_container_width=True)
    with col3:
        st.image("ukuran3.jpg", caption="3 - Seragam", use_container_width=True)

    st.subheader("Keutuhan")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("keutuhan1.jpg", caption="1 - Tidak Utuh", use_container_width=True)
    with col2:
        st.image("keutuhan2.jpg", caption="2 - Cukup Utuh", use_container_width=True)
    with col3:
        st.image("keutuhan3.jpg", caption="3 - Utuh", use_container_width=True)

    st.subheader("Kebersihan")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("kebersihan1.jpg", caption="1 - Kotor", use_container_width=True)
    with col2:
        st.image("kebersihan2.jpg", caption="2 - Cukup Bersih", use_container_width=True)
    with col3:
        st.image("kebersihan3.jpg", caption="3 - Bersih", use_container_width=True)

    st.markdown("### 📝 Pemberian Nilai Pada Beras Anda:")
    st.markdown("""
    <div style='background-color:#f9f9f9; padding:20px; border-radius:12px; margin-top:20px; color: black;'>
       <p style='text-align: center; font-size: 18px;'>
        <b>Pada bagian ini, Anda diminta untuk memberikan penilaian terhadap beras yang akan diklasifikasikan berdasarkan pengamatan langsung. Silakan isi nilai untuk setiap parameter fisik, yaitu warna, ukuran, keutuhan, dan kebersihan, dengan skala 1 hingga 3. Nilai 1 menunjukkan kualitas rendah, 2 untuk kualitas sedang, dan 3 untuk kualitas tinggi.</b>
    </p>
    <p style='text-align: center; font-size: 18px;'>
        <b>Anda dapat mengetikkan nilai secara langsung pada kolom yang tersedia, atau menggunakan tombol plus (+) dan minus (–) untuk menyesuaikan nilai. Setelah seluruh parameter dinilai, tekan tombol "Klasifikasikan" untuk melihat hasil klasifikasi kualitas beras berdasarkan input Anda.</b>
    </p>      
    """, unsafe_allow_html=True)              
                
    warna = st.number_input("Warna (1-3)", 1, 3, step=1, format="%d")
    ukuran = st.number_input("Ukuran (1-3)", 1, 3, step=1, format="%d")
    keutuhan = st.number_input("Keutuhan (1-3)", 1, 3, step=1, format="%d")
    kebersihan = st.number_input("Kebersihan (1-3)", 1, 3, step=1, format="%d")

    if st.button("🔍 Klasifikasikan", use_container_width=True):
        
        input_data = [[warna, ukuran, keutuhan, kebersihan]]
        pred = model.predict(input_data)[0]
        hasil = label_decode.get(pred, "Tidak diketahui")

        st.subheader("📌 Hasil Klasifikasi")

        st.markdown("#### 📥 Parameter yang Dimasukkan:")
        st.write(f"- Warna: {warna}")
        st.write(f"- Ukuran: {ukuran}")
        st.write(f"- Keutuhan: {keutuhan}")
        st.write(f"- Kebersihan: {kebersihan}")

        st.markdown("#### Kategori Kualitas Beras:")
        if hasil == "Premium":
            st.markdown(f"""
            <div style="margin-top:20px; background-color: #e6f4ea; border-radius: 10px 20px; color: black;">
            <h4 style="text-align: center;">
            ✅ Kategori Kualitas Beras: 
            <span style="color:green; font-weight:bold;">{hasil}</span>
            <p>Layak untuk dikonsumsi.</p>
            </h4>
            </div>
            """, unsafe_allow_html=True)

        elif hasil == "Medium":
            st.markdown(f"""
            <div style="margin-top:20px; background-color:  #FFDE21; border-radius: 10px 20px; color: black;">
            <h4 style="text-align: center;">
            ✅ Kategori Kualitas Beras: 
            <span style="color:orange; font-weight:bold;">{hasil}</span>
            <p>Bisa dikonsumsi, tapi kualitasnya sedang. Coba bersihkan atau olah lagi agar hasilnya lebih baik.</p>
            </h4>
            </div>
            """, unsafe_allow_html=True)
        elif hasil == "Rendah":
            st.markdown(f"""
            <div style="margin-top:20px; background-color: #FF0000; border-radius: 10px 20px; color: black;">
            <h4 style="text-align: center;">
            ✅ Kategori Kualitas Beras: 
            <span style="color:black; font-weight:bold;">{hasil}</span>
            <p>Kualitas kurang baik untuk konsumsi. Sebaiknya diproses lebih lanjut atau digunakan sebagai bahan campuran.</p>
            </h4>
            </div>
            """, unsafe_allow_html=True)
