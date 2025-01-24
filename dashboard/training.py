import streamlit as st
from PIL import Image
from data_ml_assignment.constants import CM_PLOT_PATH
from data_ml_assignment.processing.processing_data import ProcessingData
from data_ml_assignment.feature_engineering.feature_engineering_data import FeatureEngineeringData
from data_ml_assignment.training.train_pipeline import TrainingPipeline

def render_training():
    """
    Render the training pipeline interface in the Streamlit app.
    """
    st.header("Pipeline Training")
    st.info(
        "This pipeline consists of three steps: Processing, Feature Engineering, and Training. "
        "You can execute each step individually using the buttons below."
    )

    # Step 1: Processing Data
    st.subheader("Step 1: Process Data")
    if st.button("Process Data"):
        with st.spinner("Processing data, please wait..."):
            try:
                processor = ProcessingData()
                processor.process()
                st.success("Data processing completed successfully!")
            except Exception as e:
                st.error("Failed to process data!")
                st.exception(e)

    # Step 2: Feature Engineering
    st.subheader("Step 2: Feature Engineering")
    if st.button("Perform Feature Engineering"):
        with st.spinner("Performing feature engineering, please wait..."):
            try:
                fe = FeatureEngineeringData()
                fe.fit()  # Fit the TF-IDF vectorizer
                X, y = fe.transform()  # Transform the data
                st.session_state["X"] = X
                st.session_state["y"] = y
                st.success("Feature engineering completed successfully!")
            except Exception as e:
                st.error("Failed to perform feature engineering!")
                st.exception(e)

    # Step 3: Train Model
    st.subheader("Step 3: Train Model")
    name = st.text_input("Pipeline name", placeholder="XGBoost")
    serialize = st.checkbox("Save pipeline")
    if st.button("Train Model"):
        if "X" not in st.session_state or "y" not in st.session_state:
            st.error("Please perform feature engineering first!")
        else:
            with st.spinner("Training model, please wait..."):
                try:
                    X = st.session_state["X"]
                    y = st.session_state["y"]
                    tp = TrainingPipeline(X, y)
                    tp.train(serialize=serialize, model_name=name)

                    # Evaluate model
                    accuracy, f1 = tp.get_model_performance()
                    st.success("Model training completed successfully!")
                    col1, col2 = st.columns(2)
                    col1.metric(label="Accuracy score", value=str(round(accuracy, 4)))
                    col2.metric(label="F1 score", value=str(round(f1, 4)))

                    # Render confusion matrix
                    tp.render_confusion_matrix()
                    st.image(Image.open(CM_PLOT_PATH), width=850)

                except Exception as e:
                    st.error("Failed to train the model!")
                    st.exception(e)