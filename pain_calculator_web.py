"""
Web version of the Pain Score Calculator using Streamlit

Run with: streamlit run pain_calculator_web.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

# Import prediction logic from pain_calculator.py
from pain_calculator import predict_pain, get_feature_descriptions

# Constants
T3_IMPORTANT_FEATURES = [
    'LOS', 'BMI_Current', 'WOMACP_5', 'WeightCurrent', 'ICOAPC_3',
    'ICOAPC_1', 'AgePreOp', 'WOMACP_3', 'WalkPain', 'MobilityAidWalker',
    'Pre-Op Pain', 'HeightCurrent', 'ResultsRelief'
]

T5_IMPORTANT_FEATURES = [
    'AgePreOp', 'BMI_Current', 'WeightCurrent', 'HeightCurrent', 'LOS',
    'WOMACP_5', 'ResultsRelief', 'ICOAPC_3', 'Pre-Op Pain', 'WalkPain',
    'Approach', 'HeadSize'
]

MODELS_DIR = 'trained_models'

# Set up page
st.set_page_config(
    page_title="Hip Replacement Pain Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Hip Replacement Pain Predictor")
    st.markdown("""
    This tool predicts post-operative pain scores for hip replacement patients at two timepoints:
    * **T3**: 6 weeks post-operation
    * **T5**: 6 months post-operation
    
    **Note**: This calculator is based on statistical models and should be used only as a reference. 
    Actual patient outcomes may vary. Always consult with healthcare professionals for medical advice.
    """)
    
    # Check if models exist
    if not check_models_exist():
        st.error("Pre-trained models not found! Please run 'train_models.py' first.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("Pain Score Prediction")
    timepoint = st.sidebar.radio("Select timepoint to predict:", ["T3 (6 weeks)", "T5 (6 months)"])
    
    # Remove parentheses from timepoint string
    timepoint_code = timepoint.split(" ")[0]
    
    st.header(f"Patient Information for {timepoint}")
    
    # Get required features based on timepoint
    required_features = T3_IMPORTANT_FEATURES if timepoint_code == "T3" else T5_IMPORTANT_FEATURES
    feature_descriptions = get_feature_descriptions()
    
    # Use columns to organize the layout
    col1, col2 = st.columns(2)
    
    # Initialize patient data dictionary
    patient_data = {}
    
    # Create input fields for required features
    for i, feature in enumerate(required_features):
        description = feature_descriptions.get(feature, "")
        
        # Decide which column to put the feature in (alternate between columns)
        current_col = col1 if i % 2 == 0 else col2
        
        with current_col:
            if feature == 'MobilityAidWalker':
                patient_data[feature] = int(st.selectbox(
                    f"{feature} ({description})",
                    options=[0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes"
                ))
            elif feature == 'Approach':
                patient_data[feature] = st.selectbox(
                    f"{feature} ({description})",
                    options=["Posterior", "Anterior", "Lateral", "Other"]
                )
            elif feature in ['WOMACP_5', 'WOMACP_3', 'ICOAPC_3', 'ICOAPC_1']:
                # WOMAC and ICOA scores are on 0-4 scale
                patient_data[feature] = st.slider(
                    f"{feature} ({description})",
                    min_value=0,
                    max_value=4,
                    step=1
                )
            elif feature == 'ResultsRelief':
                # ResultsRelief is on 1-5 scale
                patient_data[feature] = st.slider(
                    f"{feature} ({description})",
                    min_value=1,
                    max_value=5,
                    step=1
                )
            elif feature == 'WalkPain' or feature == 'Pre-Op Pain':
                # Pain scores are on 0-10 scale
                patient_data[feature] = st.slider(
                    f"{feature} ({description})",
                    min_value=0,
                    max_value=10,
                    step=1
                )
            elif feature == 'HeadSize':
                # HeadSize is typically 28, 32, or 36 mm
                patient_data[feature] = st.selectbox(
                    f"{feature} ({description})",
                    options=["28", "32", "36", "40", "Other"]
                )
            else:
                # Default numeric input for other fields
                patient_data[feature] = st.number_input(
                    f"{feature} ({description})",
                    value=0.0,
                    step=0.1
                )
    
    # Predict button
    if st.button("Predict Pain Score"):
        try:
            prediction = predict_pain(patient_data, timepoint_code)
            
            # Display results
            st.header("Prediction Results")
            
            # Create a gauge-chart-like display
            fig, ax = plt.subplots(figsize=(10, 2))
            
            # Create a color gradient for the gauge
            cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green reversed (red is high pain)
            
            # Background bar (grey)
            ax.barh(0, 8, color='lightgrey', alpha=0.3)
            
            # Colored bar based on prediction
            ax.barh(0, prediction, color=cmap(prediction/8))
            
            # Customize appearance
            ax.set_xlim(0, 8)
            ax.set_yticks([])
            ax.set_xticks([0, 2, 4, 6, 8])
            ax.set_xticklabels(['0\nNo Pain', '2', '4', '6', '8\nExtreme Pain'])
            
            # Add marker for the prediction
            ax.plot(prediction, 0, 'ko', markersize=12)
            ax.text(prediction, 0, f'{prediction:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
            
            # Remove y-axis
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # Display the plot
            st.pyplot(fig)
            
            # Interpret the prediction
            if prediction <= 2:
                interpretation = "minimal"
                color = "green"
            elif prediction <= 4:
                interpretation = "mild"
                color = "blue"
            elif prediction <= 6:
                interpretation = "moderate"
                color = "orange"
            else:
                interpretation = "severe"
                color = "red"
            
            st.markdown(f"<h3 style='color:{color}'>Predicted pain level: <b>{interpretation}</b></h3>", unsafe_allow_html=True)
            
            # Additional interpretation
            st.markdown(f"""
            This prediction suggests a **{interpretation}** pain level ({prediction:.1f}/8) at {timepoint}.
            
            **Remember**:
            - This is a statistical prediction and individual results may vary
            - The model has 40-49% accuracy within ±1 point of actual pain
            - Always consult with healthcare professionals for medical advice
            """)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def check_models_exist():
    """Check if pre-trained models exist"""
    model_files = [
        os.path.join(MODELS_DIR, 't3_model.pkl'),
        os.path.join(MODELS_DIR, 't3_preprocessor.pkl'),
        os.path.join(MODELS_DIR, 't5_model.pkl'),
        os.path.join(MODELS_DIR, 't5_preprocessor.pkl')
    ]
    
    return all(os.path.exists(f) for f in model_files)

if __name__ == "__main__":
    main()
