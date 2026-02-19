# SMART - Small Molecule Analysis & Reporting Tool

> **Uber Hackathon Finalist** 🏆

An end-to-end machine learning system that predicts the **toxicity of chemical compounds** using the [Tox21](https://tox21.gov/) benchmark dataset. Input a compound via its SMILES notation and 12 biological assay results, and SMART returns a binary toxic / non-toxic classification in real time.

**Live demo:** [mtameem-smart.streamlit.app](https://mtameem-smart.streamlit.app/)

---

## What it does

Tox21 is one of the most widely used benchmarks in computational toxicology. SMART wraps it in a deployable, interactive app:

1. **Preprocessing** - raw SMILES strings are character-level tokenised and padded; the 12 assay columns are one-hot encoded.
2. **Dual-path neural network** - one branch processes the encoded SMILES through an embedding + global average pooling; the other processes the one-hot assay features through a dense layer. Both branches are concatenated before a sigmoid output.
3. **Result** - the model outputs a score in [0, 1]; ≥ 0.5 → Toxic, < 0.5 → Not Toxic.

---

## Repo layout

```
smart-toxicity-predictor/
├── app/
│   ├── smart_app.py      # Streamlit application
│   ├── smartmodel.h5     # Trained Keras model (109 KB)
│   └── assets/           # Training curves and architecture diagram
├── data/
│   ├── tcfinalit2.csv                 # Processed Tox21 dataset (738 KB)
│   └── tox21_10k_data_all_pandas.csv  # Raw Tox21 dataset (1 MB)
├── notebooks/
│   └── smart-model-code.ipynb  # Full model training walkthrough
├── website/
│   ├── index.html        # Project landing page
│   ├── about-us.html     # Team page
│   ├── smart-proposal.pdf
│   ├── css/              # Porto theme CSS (stripped to used files only)
│   ├── vendor/           # Bootstrap, FontAwesome, owl.carousel, jQuery only
│   ├── js/
│   └── img/              # Team photos, logo, content images only
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### Docker (recommended)

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501).

### Local

```bash
pip install -r requirements.txt
streamlit run app/smart_app.py
```

---

## API keys / environment variables

**None required.** The app runs entirely on local data and a pre-trained model file. No external API calls, no `.env` file needed.

---

## Tech stack

| Layer | Technology |
|---|---|
| ML framework | TensorFlow / Keras |
| Data processing | Pandas, NumPy |
| App / UI | Streamlit |
| Container | Docker |
| Dataset | Tox21 (NIH) |

---

## Model architecture

```
SMILES input (char-level tokens, max len 342)
    └─ Embedding(vocab=56, dim=8) → GlobalAveragePooling1D → Dense(64)
                                                                       ╲
                                                                Concat(128) → Dropout(0.25) → Dense(1, sigmoid)
                                                                       ╱
Assay input (36-dim one-hot vector)
    └─ Dense(64, relu)
```

Total parameters: **6 081** - deliberately small so the model ships with the repo (`smartmodel.h5`, 109 KB).

---

## Training results

| Metric | Value |
|---|---|
| Training accuracy | 99.98 % |
| Validation accuracy | 99.99 % |
| Epochs | 10 |
| Batch size | 64 |
| Optimizer | Adam |
| Loss | Binary crossentropy |

---

## Notebook

[`notebooks/smart-model-code.ipynb`](notebooks/smart-model-code.ipynb) walks through every step end-to-end: data loading, preprocessing, model definition, training, evaluation, and export.
