import streamlit as st
from data_ml_assignment.constants import LABELS_MAP
from dashboard.dashboard_utils import getSampleText, SampleRequest

def render_inference():
    st.header("Resume Inference")
    st.info(
        "This section simplifies the inference process. "
        "Choose a test resume and observe the label that your trained pipeline will predict."
    )

    sample = st.selectbox(
        "Resume samples for inference",
        tuple(LABELS_MAP.values()),
        index=None,
        placeholder="Select a resume sample",
    )
    infer = st.button("Run Inference")

    if infer:
        with st.spinner("Running inference..."):
            try:
                sample_text = getSampleText(sample)
                result = SampleRequest(sample_text)
                print(result)  # Debugging: Print the API response

                # Extract the predicted label from the API response
                if result.status_code == 200:
                    predicted_label = result.json().get("label")
                    label = LABELS_MAP.get(predicted_label)
                    st.success("Done!")
                    st.metric(label="Status", value=f"Resume label: {label}")
                else:
                    st.error(f"API returned an error: {result.text}")
            except Exception as e:
                st.error("Failed to call Inference API!")
                st.exception(e)