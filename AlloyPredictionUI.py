import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


df = pd.read_csv("final_dataset1.csv")

features_df = df.iloc[:, 1:27]

st.title("Alloy Property Prediction App")

st.write("Enter the elements and their compositions for prediction.")

num_elements = st.number_input(
    "How many elements are in the alloy?",
    min_value=1,
    max_value=features_df.shape[1],
    value=2,
    step=1
)

element_options = list(features_df.columns)

elements = []
compositions = []

st.subheader("Enter Element Details")

for i in range(num_elements):
    col1, col2 = st.columns([2, 1])
    with col1:
        element = st.selectbox(
            f"Element {i+1}",
            options=element_options,
            key=f"element_{i}"
        )
    with col2:
        composition = st.number_input(
            f"Composition % of {element}",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            key=f"composition_{i}"
        )
    elements.append(element)
    compositions.append(composition)

if st.button("Predict Properties"):

    total_composition = sum(compositions)
    if abs(total_composition - 100.0) > 1e-3:
        st.error(f"Total composition must sum to 100%. Currently: {total_composition:.2f}%. Please adjust.")
    else:
        row_idx = 0
        input_row = features_df.iloc[row_idx].copy()
        input_row[:] = 0.0

        for idx in range(num_elements):
            element = elements[idx]
            composition = compositions[idx]
            if element in input_row.index:
                input_row[element] = composition
            else:
                st.warning(f"Element {element} not found in dataset columns. Ignored.")

        input_df = input_row.to_frame().T

        prediction = model.predict(input_df)[0]  
        with open("sc.pkl", "rb") as f:
        sc = pickle.load(f)

        prediction_original = sc.inverse_transform(prediction)

        property_names =  ["Calculated Density", "Test Temperature", "Yield Strength", "Ultimate Tensile Strength", "Elongation", "Calculated Young's Modulus"]

        prediction_df = pd.DataFrame({
            "Property": property_names,
            "Predicted Value": prediction_original
        })


        st.subheader("Predicted Properties:")
        st.write(prediction_df)
