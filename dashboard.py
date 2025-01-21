import streamlit as st
from dashboard.eda import render_eda
from dashboard.inference import render_interface
from dashboard.training import render_training

def main():
    st.title("Resume Classification Dashboard")
    st.sidebar.title("Dashboard Modes")

    sidebar_options = st.sidebar.selectbox("Options", ("EDA", "Training", "Inference"))

    if sidebar_options == "EDA":
        render_eda()
    elif sidebar_options == "Training":
        render_training()
    else:
        render_interface()
        
if __name__ == "__main__":
    main()