import matplotlib
from sqlalchemy import true
import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import random
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
# %matplotlib inline

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

tox_df = pd.read_csv('tcfinalit2.csv')

model = load_model('smartmodel.h5')

# Smiles representation and chemical tests
SMILES_COL = ["smiles",]
FEATURE_COLS = ['SR-HSE', 'NR-AR', 'SR-ARE', 'NR-Aromatase', 'NR-ER-LBD', 'NR-AhR',
                'SR-MMP', 'NR-ER', 'NR-PPAR-gamma', 'SR-p53', 'SR-ATAD5', 'NR-AR-LBD']
LABEL_COLS = ['Toxic',]


tox_df_features = pd.get_dummies(tox_df[FEATURE_COLS].astype(str))
test_shape = tox_df_features.shape[-1]

def prediction(x):
    dummy_val_test = np.expand_dims(train_x_test[x], axis=0)
    dummy_val_smiles = np.expand_dims(train_x_smiles[x], axis=0)
    cheese = tox_df["smiles"][x]
    bottles = model.predict((dummy_val_smiles, dummy_val_test,))
    st.write("Your SMILE is: \n", cheese)

    penicillin_g = Chem.MolFromSmiles(cheese)

    smile_render = Draw.MolToMPL(penicillin_g, size = (200,200), fitImage=True)
    plt.show()
    plt.savefig("SMILE_rendering.png", bbox_inches = 'tight')
    st.image(".\SMILE_rendering.png")
    st.write("The AI Model's prediction", bottles)
    st.write("SMART's Prediction is: \n")
    if(bottles >= 0.5):
        st.subheader("Toxic!")
    else :
        st.subheader("Not Toxic!")

def flatten_l_o_l(nested_list):
    """ Flatten a list of lists """
    return [item for sublist in nested_list for item in sublist]

# Figure stuff out
tox_df["smile_chars"] = tox_df["smiles"].apply(list)
tox_df["smile_len"] = tox_df["smiles"].apply(len)
max_smile_len = tox_df["smile_len"].max()
tox_charlist = tox_df["smile_chars"].to_list()
all_smile_chars = flatten_l_o_l(tox_charlist)
smile_char_counts = pd.Series(all_smile_chars).value_counts()
possible_smile_chars = list(smile_char_counts.keys())
n_smile_elems = len(possible_smile_chars)+1
smile_char_map_c2i = {_c:i+1 for i,_c in enumerate(possible_smile_chars)}
smile_char_map_c2i.update({"<PAD>":0})
smile_char_map_i2c = {i+1:_c for i,_c in enumerate(possible_smile_chars)}
smile_char_map_i2c.update({0:"<PAD>"})



def encode_smiles(smile_chars, max_len, pad_token="<PAD>", pad_int=0):
    smile_encoding = [smile_char_map_c2i[_c] for _c in smile_chars]
    smile_encoding = smile_encoding+[pad_int,]*max_len
    return smile_encoding[:max_len]

tox_df["smile_encoding"] = tox_df["smile_chars"].apply(lambda x: encode_smiles(smile_chars=x, max_len=max_smile_len))


# Define number of validation examples
n_val = 1
# n_val = 10000
val_indices = np.array(random.sample(range(len(tox_df)), n_val))
train_indices = np.array([x for x in range(len(tox_df)) if x not in val_indices])

# Get the training data as numpy arrays - (11758, N)
train_x_test = tox_df_features.to_numpy().copy()
# train_x_test[:, :16] = 0
train_x_smiles = np.array(tox_df.smile_encoding.to_list())
train_y = tox_df.Toxic.to_numpy()

# Take the first M training examples to use for validation 
val_x_test = train_x_test[val_indices]
val_x_smiles = train_x_smiles[val_indices]
val_y = train_y[val_indices]
N_VAL = len(val_y)

# Take the remaining training examples to use for training
train_x_test = train_x_test[train_indices]
train_x_smiles = train_x_smiles[train_indices]
train_y = train_y[train_indices]
N_TRAIN = len(train_y)

st.title("What is SMART?")
st.write("SMART is an AI model that uses the Tox21 dataset to train and predict for the toxicity of a chemical compound.")
st.write("This is the original dataset.")
if st.button('Show original Tox21 dataset'):
    toxicchecker = pd.read_csv('tox21_10k_data_all_pandas.csv')
    toxicchecker
    st.caption("As it is, the data isn't legible.\n")

st.write("\nAs you might have seen, this wouldn't exactly work too well for our purposes. So, using RDKIT, we modified it. Here's what we got.")
if st.button('Show modified dataset'):
    tcfinal = tcfinal = pd.read_csv('tcfinalit2.csv')
    tcfinal
    st.caption("That's a lot better!")

st.write("Now it's time to preprocess it. Our demo model uses NLP Tokenization to One-Hot encode the SMILES into something the AI model can train on and predict from.")
if st.button('Show preprocessed data'):
    tox_df = pd.read_csv('toxicdf.csv')
    tox_df

st.write("Take a look at our training metrics!")

if st.button("Show Model Results"):
    accuracy = Image.open('.\Accuracy.png')
    loss = Image.open('.\Loss.png')
    st.image(accuracy)
    st.image(loss)

st.header("Try It Yourself!")
st.write("Put one of the items in the dataset into our system.")
user_prediction = st.number_input("Input the line of the ToxDF dataset you would like the AI to predict for:", 1, 10000, 200)

user_prediction = int(user_prediction)
st.write(f"Your current input is: {user_prediction}")

if st.checkbox('ToxDF Dataset'):
    toxey = pd.read_csv('toxicdf.csv')
    toxey


if st.button("Submit your whole thing!"):
    prediction(user_prediction)

