import pandas as pd
import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))
st.title("Advertising Sales Prediction")
st.write("Enter the values to get the predicted sales")

tv=st.number_input(" EnterTV Budget", min_value=0.0, key='tv')
radio=st.number_input(" Enter Radio Budget", min_value=0.0, key='radio')
newspaper=st.number_input(" Enter Newspaper Budget", min_value=0.0, key='newspaper')


if st.button("Predict the sales"):
    input_values=pd.DataFrame(
        [[float(tv), float(radio), float(newspaper)]], columns=['TV', 'Radio', 'Newspaper']
    )
    prediction=model.predict(input_values)
    st.success(f"The predicted sales is {prediction[0]:.2f}")