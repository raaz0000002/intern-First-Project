import streamlit as st
import pandas as pd
import numpy as np

st.title("Mobile Price Prediction")

col1, col2, col3, col4 = st.columns(4)

# Numerical columns
with col1:
    battery_power = st.number_input("Battery Power")

with col2:
    clock_speed = st.number_input("Clock Speed")

with col3:
    fc = st.number_input("Front Camera (FC)")

with col4:
    int_memory = st.number_input("Internal Memory")

st.divider()

col5, col6, col7, col8 = st.columns(4)

with col5:
    m_dep = st.number_input("Mobile Depth (m_dep)")

with col6:
    mobile_wt = st.number_input("Mobile Weight")

with col7:
    n_cores = st.number_input("No. of Cores (n_cores)")

with col8:
    pc = st.number_input("Primary Camera (pc)")

st.divider()

col9, col10, col11, col12 = st.columns(4)

with col9:
    px_height = st.number_input("Pixel Height (px_height)")

with col10:
    px_width = st.number_input("Pixel Width (px_width)")

with col11:
    ram = st.number_input("RAM")

with col12:
    sc_h = st.number_input("Screen Height (sc_h)")

st.divider()

col13, col14 = st.columns(2)

with col13:
    sc_w = st.number_input("Screen Width (sc_w)")

with col14:
    talk_time = st.number_input("Talk Time (talk_time)")

st.divider()

# Categorical Columns (Binary Features)
col15, col16 = st.columns(2)

with col15:
    blue = "yes" if st.toggle("Bluetooth (Blue)") else "no"
    dual_sim = "yes" if st.toggle("Dual SIM") else "no"
    four_g = "yes" if st.toggle("Four G") else "no"

with col16:
    three_g = "yes" if st.toggle("Three G") else "no"
    touch_screen = "yes" if st.toggle("Touch Screen") else "no"
    wifi = "yes" if st.toggle("WiFi") else "no"

st.divider()

# Input Dictionary
input_values = {
    'battery_power': [battery_power],
    'blue': [blue],
    'clock_speed': [clock_speed],
    'dual_sim': [dual_sim],
    'fc': [fc],
    'four_g': [four_g],
    'int_memory': [int_memory],
    'm_dep': [m_dep],
    'mobile_wt': [mobile_wt],
    'n_cores': [n_cores],
    'pc': [pc],
    'px_height': [px_height],
    'px_width': [px_width],
    'ram': [ram],
    'sc_h': [sc_h],
    'sc_w': [sc_w],
    'talk_time': [talk_time],
    'three_g': [three_g],
    'touch_screen': [touch_screen],
    'wifi': [wifi]
}

input_df = pd.DataFrame(input_values)

# Prediction Button
if st.button("Predict"):
    from inference import inference  
    result = inference(input_df)
    st.write("Predicted Price Range:", result)
