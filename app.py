import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

# =========================
# CONFIG & INIT
# =========================
st.set_page_config(page_title="Anime Recommender", layout="wide")

@st.cache_resource
def init_connection():
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception as e:
        return None

supabase = init_connection()

if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Tambahan: Memori sementara untuk menahan hasil pencarian
if 'search_result' not in st.session_state:
    st.session_state.search_result = None

# =========================
# LOAD DATA & BULLETPROOF PREPROCESS
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("anime_reference.csv")
    df.columns = df.columns.str.lower()
    
    if "genre" not in df.columns:
        df["genre"] = ""
    df["genre"] = df["genre"].fillna("")
    
    if "score" not in df.columns:
        df["score"] = 0.0
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    
    if "members" not in df.columns:
        df["members"] = 0.0
    df["members"] = pd.to_numeric(df["members"], errors="coerce").fillna(0)
    
    df["popularity"] = (df["score"] * 0.7) + (np.log1p(df["members"]) * 0.3)
    return df

@st.cache_resource
def load_model():
    svd = joblib.load("svd_model.pkl")
    tfidf = joblib.load("tfidf_matrix.pkl")
    return svd, tfidf

# =========================
# RECOM LOGIC
# =========================
def get_recommendations(judul, anime_df, svd_model, tfidf_matrix, top_n=10):
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
                sim = np.dot(target_vec, vec) / ((np.linalg.norm(target_vec) * np.linalg.norm(vec)) + 1e-8)
                skor_svd[id_map[raw_id]] = sim
        bobot = [0.4, 0.4, 0.2]
    except:
        bobot = [0.0, 0.7, 0.3]

    skor_pop = anime_df["popularity"].values
    skor_pop = (skor_pop - skor_pop.min()) / (skor_pop.max() - skor_pop.min() + 1e-8)
    
    skor_final = (skor_svd * bobot[0]) + (skor_genre * bobot[1]) + (skor_pop * bobot[2])
    skor_final[idx] = -1
    return anime_df.iloc[skor_final.argsort()[::-1][:top_n]]

# =========================
# HELPER DATABASE
# =========================
def save_favorite(user_id, anime_id, title):
    if supabase:
        # Tambahan: Cek dulu ke Supabase apakah anime ini sudah pernah disimpan user ini
        existing = supabase.table("favorites").select("*").eq("user_id", user_id).eq("anime_id", int(anime_id)).execute()
        
        if len(existing.data) == 0:
            data = {"user_id": user_id, "anime_id": int(anime_id), "anime_title": title}
            supabase.table("favorites").insert(data).execute()
            st.toast(f"✅ {title} berhasil disimpan ke favorit!")
        else:
            st.toast(f"⚠️ {title} sudah ada di daftar favoritmu!")
    else:
        st.error("Gagal menyimpan, database tidak terhubung.")

def get_user_favorites(user_id):
    if supabase:
        res = supabase.table("favorites").select("*").eq("user_id", user_id).execute()
        return res.data
    return []

# =========================
# UI PAGES
# =========================
def login_page():
    st.title("🎌 Anime Recommendation System")
    st.markdown("Silakan Login atau Register untuk masuk ke dalam sistem.")
    
    if supabase is None:
        st.error("Koneksi Supabase belum diatur di Streamlit Secrets.")
        return

    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Masuk ke Akun")
        email_login = st.text_input("Email", key="log_email")
        pass_login = st.text_input("Password", type="password", key="log_pass")
        if st.button("Login", type="primary"):
            try:
                response = supabase.auth.sign_in_with_password({"email": email_login, "password": pass_login})
                st.session_state.user = response.user
                st.success("Login berhasil!")
                st.rerun()
            except Exception as e:
                st.error("Login gagal. Periksa kembali email dan password.")

    with tab2:
        st.subheader("Buat Akun Baru")
        email_reg = st.text_input("Email", key="reg_email")
        pass_reg = st.text_input("Password", type="password", key="reg_pass")
        if st.button("Register"):
            try:
                response = supabase.auth.sign_up({"email": email_reg, "password": pass_reg})
                st.success("Registrasi berhasil! Silakan pindah ke tab Login untuk masuk.")
            except Exception as e:
                st.error("Registrasi gagal. Pastikan password minimal 6 karakter.")

def home_page(anime_df, svd_model, tfidf_matrix):
    st.title("🔍 Cari Rekomendasi")
    selected = st.selectbox("Pilih Anime:", anime_df["title"].sort_values())
    
    # Perbaikan: Simpan hasil pencarian ke memori (session_state)
    if st.button("Cari", type="primary"):
        st.session_state.search_result = get_recommendations(selected, anime_df, svd_model, tfidf_matrix)
        
    # Render hasil dari memori agar tidak hilang saat layarnya ter-refresh
    if st.session_state.search_result is not None:
        hasil = st.session_state.search_result
        if len(hasil) > 0:
            for _, row in hasil.iterrows():
                with st.container(border=True):
                    col1, col2 = st.columns([0.8, 0.2])
                    col1.write(f"**{row['title']}**")
                    col1.caption(f"Genre: {row['genre']} | Score: {row['score']:.2f}")
                    if col2.button("⭐ Simpan", key=f"fav_{row['anime_id']}"):
                        save_favorite(st.session_state.user.id, row['anime_id'], row['title'])
        else:
            st.error("Anime tidak ditemukan.")

def profile_page(anime_df, svd_model, tfidf_matrix):
    st.title("👤 Profil Saya")
    favs = get_user_favorites(st.session_state.user.id)
    
    if not favs:
        st.info("Belum ada anime favorit yang disimpan.")
        return

    st.subheader("Anime Favorit Anda:")
    for f in favs:
        with st.expander(f"📖 {f['anime_title']}"):
            if st.button(f"🎯 Cari Rekomendasi yang Cocok", key=f"rec_{f['anime_id']}"):
                hasil = get_recommendations(f['anime_title'], anime_df, svd_model, tfidf_matrix)
                if hasil is not None:
                    st.write("Berdasarkan anime ini, kamu mungkin suka:")
                    for _, row in hasil.iterrows():
                        st.write(f"- {row['title']} (Score: {row['score']:.2f})")

# =========================
# MAIN ROUTER
# =========================
if st.session_state.user is None:
    login_page()
else:
    st.sidebar.title("Navigasi")
    if st.sidebar.button("🏠 Home"): 
        st.session_state.page = "Home"
    if st.sidebar.button("👤 Profile"): 
        st.session_state.page = "Profile"
    if st.sidebar.button("🚪 Logout"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.session_state.page = "Home"
        st.session_state.search_result = None # Bersihkan memori saat logout
        st.rerun()

    df = load_data()
    m_svd, m_tfidf = load_model()

    if st.session_state.page == "Home":
        home_page(df, m_svd, m_tfidf)
    else:
        profile_page(df, m_svd, m_tfidf)
