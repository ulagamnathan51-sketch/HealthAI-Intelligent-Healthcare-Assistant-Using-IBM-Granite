import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# 🔹 Dummy dataset load (replace with your own dataset)
@st.cache_data
def load_data():
    data = pd.read_csv("improved_disease_dataset.csv")
    return data

# 🔹 Train simple ML model for disease prediction
def train_model(data):
    encoder = LabelEncoder()
    data["disease"] = encoder.fit_transform(data["disease"])
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, encoder, acc

# 🔹 Streamlit App
def main():
    st.title("🩺 HealthAI - Intelligent Healthcare Assistant")

    menu = ["Patient Chat", "Disease Prediction", "Treatment Plans", "Health Analytics"]
    choice = st.sidebar.selectbox("Choose a feature", menu)

    data = load_data()
    model, encoder, acc = train_model(data)

    if choice == "Patient Chat":
        st.subheader("💬 Chat with AI")
        user_input = st.text_input("Ask a health-related question:")
        if st.button("Get Answer"):
            st.write("🤖 AI: (This is where IBM Watson/GPT integration will reply)")

    elif choice == "Disease Prediction":
        st.subheader("🔍 Predict Disease from Symptoms")
        symptoms = st.multiselect("Select Symptoms:", data.columns[:-1])
        if st.button("Predict"):
            input_data = np.zeros((1, data.shape[1]-1))
            for s in symptoms:
                if s in data.columns:
                    input_data[0, list(data.columns[:-1]).index(s)] = 1
            prediction = model.predict(input_data)
            st.success(f"Predicted Disease: {encoder.inverse_transform(prediction)[0]}")

    elif choice == "Treatment Plans":
        st.subheader("💊 Generate Treatment Plan")
        condition = st.text_input("Enter diagnosed condition:")
        if st.button("Generate Plan"):
            st.write(f"📋 Treatment plan for **{condition}**:")
            st.write("- Suggested Medications")
            st.write("- Lifestyle Modifications")
            st.write("- Follow-up Testing")

    elif choice == "Health Analytics":
        st.subheader("📊 Health Analytics Dashboard")
        fig = px.histogram(data, x="disease", title="Disease Distribution")
        st.plotly_chart(fig)

    st.sidebar.info(f"Model Accuracy: {acc:.2f}")

if __name__ == '__main__':
    main()