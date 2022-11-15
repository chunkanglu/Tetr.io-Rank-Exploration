import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from getData import transform_data
from getRankIMG import rank_img

st.title("Tetr.io Rank Predictor")

st.write("Model is based off data from Februrary 17, 2021")


model = pickle.load(open("Tetrio_Model.sav", 'rb'))

data = pd.read_csv("../TetrioAll.csv")
data.drop("Unnamed: 0", axis=1, inplace=True)

model_data = data.loc[:,"gamesplayed":"vs"]

display_data = st.checkbox("Display Raw Data")

if (display_data):
    st.write(data)

st.subheader("Rank Predictor")

apm = st.number_input("Enter your APM (Attack Per Minute): ", min_value=0.0, step=0.01)
pps = st.number_input("Enter your PPS (Pieces Per Second):", min_value=0.0, step=0.01)
vs = st.number_input("Enter your VS Score:", min_value=0.0, step=0.01)

if st.button("Click to calculate rank"):
    model_output = model.predict([[apm, pps, vs]])

    output = model_output[0]
    
    print(output)

    img = rank_img(output)

    st.image(img, use_column_width=True)
    