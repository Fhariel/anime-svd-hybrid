import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Anime Recommender", layout="centered")

st.title("🎌 Anime Recommendation System")
st.write("Hybrid: SVD + Genre + Popularity")

# =========================
# LOAD DATA
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

anime_df = load_data()
svd_model, tfidf_matrix = load_model()

# =========================
# PREPROCESS
# =========================
anime_df["genre"] = anime_df["genre"].fillna("")
anime_df["score"] = pd.to_numeric(anime_df.get("score", 0), errors="coerce")
anime_df["members"] = pd.to_numeric(anime_df.get("members", 0), errors="coerce")

anime_df["popularity"] = (
    anime_df["score"].fillna(0) * 0.7 +
    np.log1p(anime_df["members"].fillna(0)) * 0.3
)

# =========================
# HYBRID FUNCTION
# =========================
def rekomendasi(judul, top_n=10):

    if judul not in anime_df["title"].values:
        return None

    idx = anime_df.index[anime_df["title"] == judul][0]
    anime_id = anime_df.loc[idx, "anime_id"]

    # GENRE
    skor_genre = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    # SVD
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
                    np.linalg.norm(target_vec) * np.linalg.norm(vec)
                )
                skor_svd[idx_df] = sim

        bobot_svd = 0.4
        bobot_genre = 0.4
        bobot_pop = 0.2

    except:
        bobot_svd = 0.0
        bobot_genre = 0.7
        bobot_pop = 0.3

    # POPULARITY
    skor_pop = anime_df["popularity"].values
    skor_pop = (skor_pop - skor_pop.min()) / (skor_pop.max() - skor_pop.min() + 1e-8)

    # FINAL
    skor_final = (
        skor_svd * bobot_svd +
        skor_genre * bobot_genre +
        skor_pop * bobot_pop
    )

    skor_final[idx] = -1
    top_idx = skor_final.argsort()[::-1][:top_n]

    return anime_df.iloc[top_idx]

# =========================
# UI INPUT
# =========================
anime_list = anime_df["title"].sort_values().unique()

selected = st.selectbox("Pilih Anime:", anime_list)

if st.button("Cari Rekomendasi"):
    hasil = rekomendasi(selected)

    if hasil is None:
        st.error("Anime tidak ditemukan")
    else:
        st.subheader("🔥 Rekomendasi:")

        for _, row in hasil.iterrows():
            st.write(f"**{row['title']}**")
            st.caption(row["genre"])