import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from keras.models import load_model

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

MODEL_PATH       = os.path.join(BASE_DIR, "smartmodel.h5")
TOX21_RAW_PATH   = os.path.join(DATA_DIR, "tox21_10k_data_all_pandas.csv")
TOX21_CLEAN_PATH = os.path.join(DATA_DIR, "tcfinalit2.csv")

# ── Model ─────────────────────────────────────────────────────────────────────
model = load_model(MODEL_PATH)

np.set_printoptions(precision=3, suppress=True)

# ── Data ──────────────────────────────────────────────────────────────────────
tox_df = pd.read_csv(TOX21_CLEAN_PATH)

FEATURE_COLS = [
    "SR-HSE", "NR-AR", "SR-ARE", "NR-Aromatase", "NR-ER-LBD", "NR-AhR",
    "SR-MMP", "NR-ER", "NR-PPAR-gamma", "SR-p53", "SR-ATAD5", "NR-AR-LBD",
]

tox_df_features = pd.get_dummies(tox_df[FEATURE_COLS].astype(str))


# ── SMILES encoding ───────────────────────────────────────────────────────────
def flatten_l_o_l(nested_list):
    return [item for sublist in nested_list for item in sublist]


tox_df["smile_chars"] = tox_df["smiles"].apply(list)
tox_df["smile_len"]   = tox_df["smiles"].apply(len)
max_smile_len         = tox_df["smile_len"].max()

all_smile_chars    = flatten_l_o_l(tox_df["smile_chars"].to_list())
possible_chars     = list(pd.Series(all_smile_chars).value_counts().keys())
smile_char_map_c2i = {c: i + 1 for i, c in enumerate(possible_chars)}
smile_char_map_c2i["<PAD>"] = 0


def encode_smiles(smile_chars, max_len, pad_int=0):
    encoded = [smile_char_map_c2i[c] for c in smile_chars]
    encoded += [pad_int] * max_len
    return encoded[:max_len]


tox_df["smile_encoding"] = tox_df["smile_chars"].apply(
    lambda x: encode_smiles(x, max_smile_len)
)

train_x_test   = tox_df_features.to_numpy().copy()
train_x_smiles = np.array(tox_df.smile_encoding.to_list())
train_y        = tox_df.Toxic.to_numpy()


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_row(row_idx: int):
    smiles_input = np.expand_dims(train_x_smiles[row_idx], axis=0)
    test_input   = np.expand_dims(train_x_test[row_idx], axis=0)
    score        = model.predict((smiles_input, test_input))[0][0]

    st.write(f"**SMILES:** `{tox_df['smiles'][row_idx]}`")
    st.write(f"**Raw model output:** `{score:.4f}`")
    st.subheader("SMART's Prediction:")
    if score >= 0.5:
        st.error("⚠️  Toxic")
    else:
        st.success("✅  Not Toxic")


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SMART – Molecular Toxicity Predictor", page_icon="🧪")

st.title("🧪 SMART")
st.subheader("Small Molecule Analysis & Reporting Tool")
st.write(
    "SMART uses the [Tox21 dataset](https://tox21.gov/) and a dual-path neural network "
    "to predict whether a chemical compound is toxic, based on its SMILES notation "
    "and 12 biological assay results."
)

st.divider()

# Dataset exploration
with st.expander("📊 Explore the data"):
    if st.button("Show raw Tox21 dataset"):
        st.dataframe(pd.read_csv(TOX21_RAW_PATH))
        st.caption("Raw dataset – not yet clean enough for modelling.")

    if st.button("Show processed dataset"):
        st.dataframe(tox_df)
        st.caption("After RDKIT-based preprocessing.")

# Model details
with st.expander("🏗️ Model architecture & training results"):
    col1, col2, col3 = st.columns(3)
    col1.image(Image.open(os.path.join(ASSETS_DIR, "model.png")),    caption="Architecture")
    col2.image(Image.open(os.path.join(ASSETS_DIR, "accuracy.png")), caption="Training accuracy")
    col3.image(Image.open(os.path.join(ASSETS_DIR, "loss.png")),     caption="Training loss")

st.divider()

# Interactive prediction
st.header("Try It Yourself")
st.write(
    "Pick any row index from the processed dataset (0 – 11 757) and the model will "
    "classify that compound as toxic or not."
)

row_idx = st.number_input("Dataset row index", min_value=0, max_value=len(tox_df) - 1, value=200)

if st.button("Run prediction"):
    predict_row(int(row_idx))
