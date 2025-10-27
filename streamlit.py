import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"
st.title("Rock or Mine predictor")
st.markdown("Enter the frequencies below")

Freq_1 = st.number_input("Freq_1")
Freq_2 = st.number_input("Freq_2")
Freq_3 = st.number_input("Freq_3")
Freq_4 = st.number_input("Freq_4")
Freq_5 = st.number_input("Freq_5")
Freq_6 = st.number_input("Freq_6")
Freq_7 = st.number_input("Freq_7")
Freq_8 = st.number_input("Freq_8")
Freq_9 = st.number_input("Freq_9")
Freq_10 = st.number_input("Freq_10")
Freq_11 = st.number_input("Freq_11")
Freq_12 = st.number_input("Freq_12")
Freq_13 = st.number_input("Freq_13")
Freq_14 = st.number_input("Freq_14")
Freq_15 = st.number_input("Freq_15")
Freq_16 = st.number_input("Freq_16")
Freq_17 = st.number_input("Freq_17")
Freq_18 = st.number_input("Freq_18")
Freq_19 = st.number_input("Freq_19")
Freq_20 = st.number_input("Freq_20")
Freq_21 = st.number_input("Freq_21")
Freq_22 = st.number_input("Freq_22")
Freq_23 = st.number_input("Freq_23")
Freq_24 = st.number_input("Freq_24")
Freq_25 = st.number_input("Freq_25")
Freq_26 = st.number_input("Freq_26")
Freq_27 = st.number_input("Freq_27")
Freq_28 = st.number_input("Freq_28")
Freq_29 = st.number_input("Freq_29")
Freq_30 = st.number_input("Freq_30")
Freq_31 = st.number_input("Freq_31")
Freq_32 = st.number_input("Freq_32")
Freq_33 = st.number_input("Freq_33")
Freq_34 = st.number_input("Freq_34")
Freq_35 = st.number_input("Freq_35")
Freq_36 = st.number_input("Freq_36")
Freq_37 = st.number_input("Freq_37")
Freq_38 = st.number_input("Freq_38")
Freq_39 = st.number_input("Freq_39")
Freq_40 = st.number_input("Freq_40")
Freq_41 = st.number_input("Freq_41")
Freq_42 = st.number_input("Freq_42")
Freq_43 = st.number_input("Freq_43")
Freq_44 = st.number_input("Freq_44")
Freq_45 = st.number_input("Freq_45")
Freq_46 = st.number_input("Freq_46")
Freq_47 = st.number_input("Freq_47")
Freq_48 = st.number_input("Freq_48")
Freq_49 = st.number_input("Freq_49")
Freq_50 = st.number_input("Freq_50")
Freq_51 = st.number_input("Freq_51")
Freq_52 = st.number_input("Freq_52")
Freq_53 = st.number_input("Freq_53")
Freq_54 = st.number_input("Freq_54")
Freq_55 = st.number_input("Freq_55")
Freq_56 = st.number_input("Freq_56")
Freq_57 = st.number_input("Freq_57")
Freq_58 = st.number_input("Freq_58")
Freq_59 = st.number_input("Freq_59")
Freq_60 = st.number_input("Freq_60")


if st.button("Predict"):
    input_data = {
        "Freq_1": Freq_1,
        "Freq_2": Freq_2,
        "Freq_3": Freq_3,
        "Freq_4": Freq_4,
        "Freq_5": Freq_5,
        "Freq_6": Freq_6,
        "Freq_7": Freq_7,
        "Freq_8": Freq_8,
        "Freq_9": Freq_9,
        "Freq_10": Freq_10,
        "Freq_11": Freq_11,
        "Freq_12": Freq_12,
        "Freq_13": Freq_13,
        "Freq_14": Freq_14,
        "Freq_15": Freq_15,
        "Freq_16": Freq_16,
        "Freq_17": Freq_17,
        "Freq_18": Freq_18,
        "Freq_19": Freq_19,
        "Freq_20": Freq_20,
        "Freq_21": Freq_21,
        "Freq_22": Freq_22,
        "Freq_23": Freq_23,
        "Freq_24": Freq_24,
        "Freq_25": Freq_25,
        "Freq_26": Freq_26,
        "Freq_27": Freq_27,
        "Freq_28": Freq_28,
        "Freq_29": Freq_29,
        "Freq_30": Freq_30,
        "Freq_31": Freq_31,
        "Freq_32": Freq_32,
        "Freq_33": Freq_33,
        "Freq_34": Freq_34,
        "Freq_35": Freq_35,
        "Freq_36": Freq_36,
        "Freq_37": Freq_37,
        "Freq_38": Freq_38,
        "Freq_39": Freq_39,
        "Freq_40": Freq_40,
        "Freq_41": Freq_41,
        "Freq_42": Freq_42,
        "Freq_43": Freq_43,
        "Freq_44": Freq_44,
        "Freq_45": Freq_45,
        "Freq_46": Freq_46,
        "Freq_47": Freq_47,
        "Freq_48": Freq_48,
        "Freq_49": Freq_49,
        "Freq_50": Freq_50,
        "Freq_51": Freq_51,
        "Freq_52": Freq_52,
        "Freq_53": Freq_53,
        "Freq_54": Freq_54,
        "Freq_55": Freq_55,
        "Freq_56": Freq_56,
        "Freq_57": Freq_57,
        "Freq_58": Freq_58,
        "Freq_59": Freq_59,
        "Freq_60": Freq_60,
    }

    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"it is predicted to be a {result['prediction'][0]}")

    except requests.exceptions.ConnectionError:
        st.error("Error")
