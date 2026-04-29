import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client

# =========================
# CONFIG & INIT
# =========================
st.set_page_config(page_title="Anime Recommender", layout="centered")

# Inisialisasi koneksi Supabase menggunakan Streamlit Secrets
@st.cache_resource
def init_connection():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except:
        return None

supabase = init_connection()

# Inisialisasi Session State untuk menyimpan data user yang sedang login
if 'user' not in st.session_state:
    st.session_state.user = None

# =========================
# LOAD DATA & MODEL (SVD)
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("anime_reference.csv")
    return df

@st.cache_resource
def load_model():
    svd = joblib.load("svd_model.pkl")
    tfidf = joblib.load("tfidf_matrix.pkl")
    return svd, tfidf

# =========================
# HALAMAN LOGIN & REGISTER
# =========================
def login_page():
    st.title("🎌 Anime Recommendation System")
    st.markdown("Silakan Login atau Register untuk masuk ke dalam sistem.")
    
    if supabase is None:
        st.error("Koneksi Supabase belum diatur di Streamlit Secrets.")
        return

    tab1, tab2 = st.tabs(["Login", "Register"])
    
    # TAB LOGIN
    with tab1:
        st.subheader("Masuk ke Akun")
        email_login = st.text_input("Email", key="log_email")
        pass_login = st.text_input("Password", type="password", key="log_pass")
        
        if st.button("Login", type="primary"):
            try:
                response = supabase.auth.sign_in_with_password({"email": email_login, "password": pass_login})
                st.session_state.user = response.user
                st.success("Login berhasil!")
                st.rerun() # Refresh halaman untuk masuk ke main app
            except Exception as e:
                st.error("Login gagal. Periksa kembali email dan password.")

    # TAB REGISTER
    with tab2:
        st.subheader("Buat Akun Baru")
        email_reg = st.text_input("Email", key="reg_email")
        pass_reg = st.text_input("Password", type="password", key="reg_pass")
        
        if st.button("Register"):
            try:
                # Membuat akun baru di Supabase
                response = supabase.auth.sign_up({"email": email_reg, "password": pass_reg})
                st.success("Registrasi berhasil! Silakan pindah ke tab Login untuk masuk.")
            except Exception as e:
                st.error("Registrasi gagal. Email mungkin sudah terdaftar atau password terlalu pendek (minimal 6 karakter).")

# =========================
# HALAMAN UTAMA (REKOMENDASI)
# =========================
def main_app():
    # Tombol Logout di Pojok Kanan Atas
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.write(f"Selamat datang, **{st.session_state.user.email}**! 👋")
    with col2:
        if st.button("Logout"):
            supabase.auth.sign_out()
            st.session_state.user = None
            st.rerun()

    st.title("✨ Temukan Anime Baru")
    st.write("Sistem Cerdas Hybrid: SVD + Genre + Popularity")
    st.divider()

    # Load Data
    anime_df = load_data()
    svd_model, tfidf_matrix = load_model()

    # Preprocess
    anime_df["genre"] = anime_df["genre"].fillna("")
    anime_df["score"] = pd.to_numeric(anime_df.get("score", 0), errors="coerce")
    anime_df["members"] = pd.to_numeric(anime_df.get("members", 0), errors="coerce")
    anime_df["popularity"] = (
        anime_df["score"].fillna(0) * 0.7 +
        np.log1p(anime_df["members"].fillna(0)) * 0.3
    )

    # Fungsi Rekomendasi
    def rekomendasi(judul, top_n=10):
        if judul not in anime_df["title"].values:
            return None

        idx = anime_df.index[anime_df["title"] == judul][0]
        anime_id = int(anime_df.loc[idx, "anime_id"])

        skor_genre = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        skor_svd = np.zeros(len(anime_df))
        id_map = {aid: i for i, aid in enumerate(anime_df["anime_id"])}

        try:
            inner_id = svd_model.trainset.to_inner_iid(anime_id)
            target_vec = svd_model.qi[inner_id]

            for i, vec in enumerate(svd_model.qi):
                raw_id = svd_model.trainset.to_raw_iid(i)
                if raw_id in id_map:
                    idx_df = id_map[raw_id]
                    sim = np.dot(target_vec, vec) / (
                        (np.linalg.norm(target_vec) * np.linalg.norm(vec)) + 1e-8
                    )
                    skor_svd[idx_df] = sim

            bobot_svd = 0.4
            bobot_genre = 0.4
            bobot_pop = 0.2
        except ValueError:
            bobot_svd = 0.0
            bobot_genre = 0.7
            bobot_pop = 0.3

        skor_pop = anime_df["popularity"].values
        skor_pop = (skor_pop - skor_pop.min()) / (skor_pop.max() - skor_pop.min() + 1e-8)

        skor_final = (skor_svd * bobot_svd) + (skor_genre * bobot_genre) + (skor_pop * bobot_pop)
        skor_final[idx] = -1 
        top_idx = skor_final.argsort()[::-1][:top_n]

        return anime_df.iloc[top_idx]

    # UI Input
    anime_list = anime_df["title"].sort_values().unique()
    selected = st.selectbox("Pilih Anime yang kamu suka:", anime_list)

    if st.button("Cari Rekomendasi", type="primary"):
        with st.spinner("Mesin SVD sedang berfikir..."):
            hasil = rekomendasi(selected)

        if hasil is None:
            st.error("Anime tidak ditemukan")
        else:
            st.subheader("🔥 Rekomendasi untukmu:")
            for _, row in hasil.iterrows():
                with st.container(border=True):
                    st.write(f"**{row['title']}**")
                    st.caption(f"Genre: {row['genre']} | Score: {row['score']}")

# =========================
# ROUTER
# =========================
# Jika user belum login, tampilkan halaman login. Jika sudah, tampilkan aplikasi utama.
if st.session_state.user is None:
    login_page()
else:
    main_app()
