import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from predict import (
    load_model,
    predict_image
)

from gradcam import (
    generate_gradcam
)

from radiomics import get_top_biomarkers

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="OralAI - Explainable Oral Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#F8FAFC;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:4rem;
    padding-right:4rem;
}

h1{
    color:#0F172A;
}

h2{
    color:#1E3A8A;
}

h3{
    color:#2563EB;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    border:1px solid #E5E7EB;
    box-shadow:0px 3px 10px rgba(0,0,0,0.08);
}

.stButton>button{
    width:100%;
    background:#2563EB;
    color:white;
    border-radius:10px;
    height:3.2em;
    font-size:17px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1D4ED8;
}

.upload-box{
    background:white;
    border-radius:12px;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def get_model():

    return load_model()

try:

    model = get_model()

    MODEL_LOADED = True

except Exception as e:

    MODEL_LOADED = False

    MODEL_ERROR = str(e)

# ==========================================================
# MAIN HEADER
# ==========================================================

st.title("🩺 Oral Cancer Detection")

st.subheader(
    "Explainable Artificial Intelligence Platform for Oral Cancer Detection"
)

st.markdown("""
This platform automatically analyzes **histopathological oral tissue images**
using a fine-tuned **ResNet-50 Deep Learning Model**.

The system predicts whether the uploaded tissue image belongs to:

- 🟢 **Normal Oral Tissue**

- 🔴 **Oral Squamous Cell Carcinoma (OSCC)**

In addition to prediction, the platform provides:

- 🔥 Grad-CAM Visual Explanation

- 📊 Radiomics Biomarker Analysis

to improve model transparency and clinical interpretability.
""")

st.divider()

# ==========================================================
# MODEL STATUS
# ==========================================================

if MODEL_LOADED:

    st.success(
        "✅ AI Model Loaded Successfully and Ready for Prediction"
    )

else:

    st.error("Model Loading Failed")

    st.code(MODEL_ERROR)

    st.stop()

# ==========================================================
# IMAGE UPLOAD
# ==========================================================

st.subheader("📤 Upload Histopathological Image")

uploaded_file = st.file_uploader(
    "Choose a Histopathological Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================================
# WAIT FOR IMAGE
# ==========================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a histopathological image to begin AI analysis."
    )

else:

    image = Image.open(uploaded_file).convert("RGB")

    # ======================================================
    # IMAGE PREVIEW
    # ======================================================

    col1, col2 = st.columns([1.5, 1])

    with col1:

        st.image(
            image,
            caption="Uploaded Histopathological Image",
            use_container_width=True
        )

    with col2:

        st.subheader("📋 Image Information")

        st.metric(
            "Filename",
            uploaded_file.name
        )

        st.metric(
            "Resolution",
            f"{image.size[0]} × {image.size[1]} px"
        )

        st.metric(
            "Color Mode",
            image.mode
        )

        st.metric(
            "Image Format",
            image.format if image.format else "Unknown"
        )

        st.success("Image uploaded successfully.")

    st.divider()

    # ======================================================
    # ANALYZE BUTTON
    # ======================================================

    st.subheader("🤖 Deep Learning Diagnosis")

    analyze = st.button(
        "🔍 Analyze Image",
        use_container_width=True
    )

    # ======================================================
    # RUN MODEL
    # ======================================================

    if analyze:

        import time

        start_time = time.time()

        with st.spinner("Running ResNet-50 inference..."):

            try:

                result = predict_image(
                    model=model,
                    image=image
                )

            except Exception as e:

                st.error("Prediction Failed")

                st.exception(e)

                st.stop()

        end_time = time.time()

        processing_time = end_time - start_time

        prediction = result["class_name"]

        confidence = result["confidence"]

        probabilities = result["probabilities"]

        normal_probability = float(probabilities[0])

        oscc_probability = float(probabilities[1])

        st.divider()

        # ==================================================
        # DIAGNOSS
        # ==================================================

        st.subheader("🩺 AI Diagnosis")

        diagnosis_col, metric_col = st.columns([1.4, 1])

        with diagnosis_col:

            if prediction == "Normal":

                st.success(
                    """
### 🟢 Normal Tissue

The uploaded histopathological image has been classified as
**Normal Oral Tissue** by the AI model.

No malignant tissue patterns were detected.
"""
                )

            else:

                st.error(
                    """
### 🔴 Oral Squamous Cell Carcinoma (OSCC)

The uploaded histopathological image has been classified as
**Oral Squamous Cell Carcinoma (OSCC)**.

Abnormal tissue characteristics were detected by the AI model.
"""
                )

        with metric_col:

            st.metric(
                "Prediction",
                prediction
            )

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.metric(
                "Processing Time",
                f"{processing_time:.2f} sec"
            )

            if confidence >= 0.95:

                st.success("Confidence Level : Very High")

            elif confidence >= 0.85:

                st.info("Confidence Level : High")

            elif confidence >= 0.70:

                st.warning("Confidence Level : Moderate")

            else:

                st.error("Confidence Level : Low")

        st.divider()

        # ==================================================
        # PROBABILITY SCORES
        # ==================================================

        st.subheader("📊 Prediction Probabilities")

        prob_col1, prob_col2 = st.columns(2)

        with prob_col1:

            st.metric(
                "🟢 Normal",
                f"{normal_probability*100:.2f}%"
            )

            st.progress(normal_probability)

        with prob_col2:

            st.metric(
                "🔴 OSCC",
                f"{oscc_probability*100:.2f}%"
            )

            st.progress(oscc_probability)

        st.divider()

        # ==================================================
        # MODEL SUMMARY
        # ==================================================

        st.subheader("📌 Prediction Summary")

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Prediction",
            prediction
        )

        s2.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        s3.metric(
            "Model",
            "ResNet-50"
        )

        s4.metric(
            "Inference",
            f"{processing_time:.2f}s"
        )

        # ==========================================================
        # GRAD-CAM VISUALIZATION
        # ==========================================================

        st.divider()

        st.header("🔥 Grad-CAM Explainability")

        st.markdown(
        """
Grad-CAM (Gradient-weighted Class Activation Mapping) highlights
the image regions that contributed most to the deep learning model's
decision.

Red regions indicate stronger influence on the prediction,
whereas blue regions indicate lower influence.
"""
        )

        with st.spinner("Generating Grad-CAM Visualization..."):

            try:

                original_image, heatmap_image, overlay_image = generate_gradcam(
                    model=model,
                    image=image
                )

            except Exception as e:

                st.error("Grad-CAM generation failed.")

                st.exception(e)

                st.stop()

        original_col, heatmap_col, overlay_col = st.columns(3)

        with original_col:

            st.image(
                original_image,
                caption="Original Image",
                use_container_width=True
            )

        with heatmap_col:

            st.image(
                heatmap_image,
                caption="Grad-CAM Heatmap",
                use_container_width=True
            )

        with overlay_col:

            st.image(
                overlay_image,
                caption="Overlay Visualization",
                use_container_width=True
            )
        # BIOMARKER
        
        st.divider()

        st.subheader("🧬 Top Radiomics Biomarkers")

        try:

            biomarkers = get_top_biomarkers()

        except Exception as e:

            st.error("Unable to load radiomics biomarkers.")

            st.exception(e)

            st.stop()

        st.dataframe(

            biomarkers,

            use_container_width=True,

            hide_index=True
        )
        st.subheader("📈 Biomarker Importance Visualization")
        fig, ax = plt.subplots(figsize=(10,6))

        ax.barh(

            biomarkers["Feature"],

            biomarkers["Importance"]

        )

        ax.set_xlabel("Feature Importance")

        ax.set_ylabel("Radiomic Biomarker")

        ax.set_title("Top 10 Radiomics Biomarkers")

        ax.invert_yaxis()

        plt.tight_layout()

        st.pyplot(fig)
        st.success(
"""
Higher bars indicate radiomic features that contributed more strongly
to the XGBoost classifier.

These biomarkers were ranked using the trained model's feature importance scores.
"""
)
        st.info(
"""
### 📖 Interpretation

The biomarkers shown above represent the **10 most influential radiomic features**
identified by the trained **XGBoost classifier**.

These handcrafted imaging biomarkers contributed the most toward distinguishing
**Normal tissue** from **Oral Squamous Cell Carcinoma (OSCC)** during model training.

Higher importance values indicate a greater contribution to the classification process.
"""
)
            
# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style="text-align:center; padding:20px;">

<h2>🩺 Oral Cancer Detection</h2>

<p>
<b>Explainable Artificial Intelligence Platform for Oral Cancer Detection</b>
</p>

<p>
Powered by
<b>PyTorch</b> •
<b>ResNet-50</b> •
<b>Grad-CAM</b> •
<b>PyRadiomics</b> •
<b>XGBoost</b> •
<b>Streamlit</b>
</p>

<p>
Developed by <b>Mohd Anas Maroof</b> as part of an <b>M.Tech Major Project</b>
</p>

<p style="color:gray;font-size:13px;">
© 2026 OralAI. All Rights Reserved.
</p>

</div>
""",
unsafe_allow_html=True
)

