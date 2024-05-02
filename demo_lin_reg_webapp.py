import streamlit as st
import pickle

# Load the pickled model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create the Streamlit web app
st.header("Streamlit demo")

st.sidebar.header("This is a web app")

X_test = st.sidebar.slider("Select X to get yhat", 0, 10, 5)

st.write("X test is:", X_test)

yhat_test = model.predict([[X_test]])

st.write("b0 is", round(model.intercept_, 3))
st.write("b1 is", round(model.coef_[0], 3))
st.write("yhat test is", yhat_test)
