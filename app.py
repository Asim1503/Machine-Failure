import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    
    return data

regressor = load_model()

def show_predicted_page():
    st.title('Machine Failure')
    st.write("""### We need some information""")
    
    air_temp = st.number_input("Air Temperature (K)")
    process_temp = st.number_input("Process Temperature (K)")
    rot_speed = st.number_input("Rotational Speed (RPM)")
    torque = st.number_input("Torque (Nm)")
    wear = st.number_input("Tool Wear (min)")
    
    done = st.button('Calculate Machine Failure Chances')

    if done:
        input_features = np.array([[air_temp, process_temp, rot_speed, torque, wear]])

        # Make predictions using the trained model
        predicted_performance = regressor.predict_proba(input_features)

        # Format the prediction result based on the data type
        formatted_prediction = f"Chances of Machine Failure: {predicted_performance[0][0] * 100:.0f}%"

        st.subheader(formatted_prediction)

show_predicted_page()
