"""
Train Pain Prediction Models

This script trains Gradient Boosting models for T3 and T5 pain prediction
and saves the trained models for use by the pain calculator.
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight

# Key features identified from importance analysis (same as in pain_calculator.py)
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

def create_preprocessor(numeric_features, categorical_features):
    """Create preprocessing pipeline"""
    transformers = []
    
    if len(numeric_features) > 0:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_features))
    
    if len(categorical_features) > 0:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

def load_and_preprocess_data(timepoint, important_features):
    """Load and preprocess data focusing on important features"""
    print(f"Processing {timepoint} data...")
    
    # Load data
    df = pd.read_excel(f'{timepoint}_coded_processed.xlsx')
    
    # Separate target
    target_col = f'{timepoint} Pain'
    y = df[target_col]
    
    # Select important features
    X = df[important_features]
    
    # Remove any rows where target is 9.0 or 10.0 (invalid)
    mask = ~y.isin([9.0, 10.0])
    X = X[mask]
    y = y[mask]
    
    # Print value distributions
    print("\nTarget value distribution:")
    value_counts = y.value_counts().sort_index()
    print(value_counts)
    
    # Calculate weights inversely proportional to class frequencies
    class_weights = dict(1/value_counts)
    sample_weights = y.map(class_weights)
    
    # Convert HeadSize to string if present
    if 'HeadSize' in X.columns:
        X['HeadSize'] = X['HeadSize'].astype(str)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    print(f"\nSelected features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    
    return X, y, numeric_features, categorical_features, sample_weights

def optimize_model(X_train, y_train, sample_weights, model_name='Gradient Boosting'):
    """Train optimized model with best parameters from previous grid search"""
    print(f"Training {model_name} model...")
    
    # Best parameters from previous optimization
    if model_name == 'Gradient Boosting':
        # These parameters are from the optimized_regression_model_v4.py results
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.1, 
            max_depth=6,
            min_samples_split=2,
            subsample=0.9,
            random_state=42
        )
    
    # Fit with sample weights
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    return model

def train_and_save_models():
    """Train models and save them along with preprocessors for later use"""
    # Create models directory if it doesn't exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    # Train T3 model
    X_t3, y_t3, num_feat_t3, cat_feat_t3, weights_t3 = load_and_preprocess_data('T3', T3_IMPORTANT_FEATURES)
    X_train_t3, X_test_t3, y_train_t3, y_test_t3, weights_train_t3, _ = train_test_split(
        X_t3, y_t3, weights_t3, test_size=0.2, random_state=42, stratify=y_t3
    )
    
    # Create and fit preprocessor for T3
    preprocessor_t3 = create_preprocessor(num_feat_t3, cat_feat_t3)
    X_train_t3_proc = preprocessor_t3.fit_transform(X_train_t3)
    
    # Train optimized model for T3
    t3_model = optimize_model(X_train_t3_proc, y_train_t3, weights_train_t3)
    
    # Save T3 model and preprocessor
    with open(os.path.join(MODELS_DIR, 't3_model.pkl'), 'wb') as f:
        pickle.dump(t3_model, f)
    with open(os.path.join(MODELS_DIR, 't3_preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor_t3, f)
    with open(os.path.join(MODELS_DIR, 't3_features.pkl'), 'wb') as f:
        pickle.dump({'numeric': num_feat_t3, 'categorical': cat_feat_t3}, f)
    
    # Train T5 model
    X_t5, y_t5, num_feat_t5, cat_feat_t5, weights_t5 = load_and_preprocess_data('T5', T5_IMPORTANT_FEATURES)
    X_train_t5, X_test_t5, y_train_t5, y_test_t5, weights_train_t5, _ = train_test_split(
        X_t5, y_t5, weights_t5, test_size=0.2, random_state=42, stratify=y_t5
    )
    
    # Create and fit preprocessor for T5
    preprocessor_t5 = create_preprocessor(num_feat_t5, cat_feat_t5)
    X_train_t5_proc = preprocessor_t5.fit_transform(X_train_t5)
    
    # Train optimized model for T5
    t5_model = optimize_model(X_train_t5_proc, y_train_t5, weights_train_t5)
    
    # Save T5 model and preprocessor
    with open(os.path.join(MODELS_DIR, 't5_model.pkl'), 'wb') as f:
        pickle.dump(t5_model, f)
    with open(os.path.join(MODELS_DIR, 't5_preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor_t5, f)
    with open(os.path.join(MODELS_DIR, 't5_features.pkl'), 'wb') as f:
        pickle.dump({'numeric': num_feat_t5, 'categorical': cat_feat_t5}, f)
    
    print("\nModels trained and saved successfully in the '{}' directory.".format(MODELS_DIR))
    print("You can now use pain_calculator.py without retraining each time.")
    
    return True

if __name__ == "__main__":
    print("\n=== TRAINING PAIN PREDICTION MODELS ===")
    print("This will train and save models for T3 and T5 pain prediction.")
    print("The process may take a few minutes...\n")
    
    train_and_save_models()
