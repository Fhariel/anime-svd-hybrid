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
    except:
        return None

supabase = init_connection()

if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = "Home" # Default page

# =========================
# DATA & RECOM LOGIC
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("anime_reference.csv") #

@st.cache_resource
def load_model():
    svd = joblib.load("svd_model.pkl") #
    tfidf = joblib.load("tfidf_matrix.pkl")
    return svd, tfidf

def get_recommendations(judul, anime_df, svd_model, tfidf_matrix, top_n=10):
    # Logika Hybrid SVD + Genre + Popularity dari kodemu sebelumnya
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
                sim = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-8)
                skor_svd[id_map[raw_id]] = sim
        bobot = [0.4, 0.4, 0.2]
    except:
        bobot = [0.0, 0.7, 0.3]

    skor_pop = anime_df["score"].fillna(0) # Sederhana untuk contoh
    skor_final = (skor_svd * bobot[0]) + (skor_genre * bobot[1]) + (skor_pop * bobot[2])
    skor_final[idx] = -1
    return anime_df.iloc[skor_final.argsort()[::-1][:top_n]]

# =========================
# HELPER DATABASE
# =========================
def save_favorite(user_id, anime_id, title):
    data = {"user_id": user_id, "anime_id": anime_id, "anime_title": title}
    supabase.table("favorites").insert(data).execute()
    st.toast(f"✅ {title} disimpan ke favorit!")

def get_user_favorites(user_id):
    res = supabase.table("favorites").select("*").eq("user_id", user_id).execute()
    return res.data

# =========================
# UI PAGES
# =========================
def home_page(anime_df, svd_model, tfidf_matrix):
    st.title("🔍 Cari Rekomendasi")
    selected = st.selectbox("Pilih Anime:", anime_df["title"].sort_values())
    
    if st.button("Cari", type="primary"):
        hasil = get_recommendations(selected, anime_df, svd_model, tfidf_matrix)
        for _, row in hasil.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([0.8, 0.2])
                col1.write(f"**{row['title']}**")
                col1.caption(f"Genre: {row['genre']}")
                if col2.button("⭐ Simpan", key=f"fav_{row['anime_id']}"):
                    save_favorite(st.session_state.user.id, row['anime_id'], row['title'])

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
                st.write("Berdasarkan anime ini, kamu mungkin suka:")
                for _, row in hasil.iterrows():
                    st.write(f"- {row['title']}")

# =========================
# MAIN ROUTER
# =========================
if st.session_state.user is None:
    # Halaman Login/Register (Gunakan kode sebelumnya)
    pass 
else:
    # Sidebar Navigasi
    st.sidebar.title("Navigasi")
    if st.sidebar.button("🏠 Home"): st.session_state.page = "Home"
    if st.sidebar.button("👤 Profile"): st.session_state.page = "Profile"
    if st.sidebar.button("🚪 Logout"):
        supabase.auth.sign_out()
        st.session_state.user = None
        st.rerun()

    df = load_data()
    m_svd, m_tfidf = load_model()

    if st.session_state.page == "Home":
        home_page(df, m_svd, m_tfidf)
    else:
        profile_page(df, m_svd, m_tfidf)
