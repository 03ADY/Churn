# app.py
# To run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import joblib
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier # Keeping XGBoost as an option/reference for permutation importance


warnings.filterwarnings("ignore")

# Enhanced page configuration
st.set_page_config(
    page_title="Advanced AI Churn Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Global styles */
    .main {
        padding-top: 2rem;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        text-align: center;
        line-height: 1.6;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }

    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9ff 0%, #e8f0fe 100%);
    }

    /* Success/Error/Warning boxes */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .error-box {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .warning-box {
        background: linear-gradient(135deg, #fdbb2d 0%, #22c1c3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        color: white;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: scale(1.05);
    }

    /* Form styling */
    .prediction-form {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }

    /* Progress bar */
    .progress-container {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Feature importance styling */
    .feature-importance {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Individual prediction styling */
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
    }

    .prediction-high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }

    .prediction-medium-risk {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        color: white;
    }

    .prediction-low-risk {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
        color: white;
    }

    /* Animation for loading */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .loading-animation {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

@st.cache_data
def create_sample_data():
    """Create sample churn data for demonstration"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        'CreditScore': np.random.randint(300, 850, n_samples),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 80, n_samples),
        'Tenure': np.random.randint(0, 10, n_samples),
        'Balance': np.random.uniform(0, 250000, n_samples),
        'EstimatedSalary': np.random.uniform(0, 200000, n_samples),
    }

    df = pd.DataFrame(data)

    # Make churn prediction more realistic
    prob_adjustments = (
        (df['Age'] > 45) * 0.15 +
        (df['Balance'] == 0) * 0.2 +
        (df['CreditScore'] < 600) * 0.15 +
        (df['Tenure'] <= 1) * 0.1
    )

    final_prob = np.clip(0.2 + prob_adjustments, 0.05, 0.8)
    df['Exited'] = np.random.binomial(1, final_prob)

    # Add dummy CustomerId, Surname, RowNumber to match expected input format if needed later
    df['RowNumber'] = np.arange(1, n_samples + 1)
    df['CustomerId'] = np.random.randint(10000000, 20000000, n_samples)
    df['Surname'] = [f"Surname{i}" for i in range(n_samples)]

    return df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle data


@st.cache_data
def load_data(uploaded_file=None):
    """
    Load data from an uploaded CSV.
    If no file is uploaded or loading fails, create sample data.
    """
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {uploaded_file.name} successfully!")
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}. Creating sample data.")

    if df is None: # If no file uploaded or uploaded file failed
        df = create_sample_data()
        st.info("📊 Using automatically generated sample data for demonstration.")

    # Drop irrelevant columns if they exist
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Validate required columns for prediction and target
    required_cols_for_prediction = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'EstimatedSalary'
    ]
    
    # Check if all prediction features are present and 'Exited' column for training is present
    if not all(col in df.columns for col in required_cols_for_prediction) or 'Exited' not in df.columns:
        missing_features = [col for col in required_cols_for_prediction if col not in df.columns]
        missing_target = "'Exited'" if 'Exited' not in df.columns else ""
        
        error_msg = f"Loaded data is missing required columns for training/prediction: "
        if missing_features:
            error_msg += f"Features: {', '.join(missing_features)}. "
        if missing_target:
            error_msg += f"Target: {missing_target}."
            
        st.warning(f"{error_msg} Falling back to sample data (if not already using it).")
        df = create_sample_data() # Force sample data if loaded data is incomplete
        st.info("📊 Using automatically generated sample data for demonstration due to incomplete data.")

    return df


@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def train_models(df):
    """Train both Random Forest and Neural Network models"""
    if df is None or 'Exited' not in df.columns:
        st.error("Dataset is invalid or missing 'Exited' column.")
        return None, None, None, None, None, None

    if len(df) < 50:
        st.error("Dataset has too few samples for meaningful training. Please provide at least 50 rows.")
        return None, None, None, None, None, None

    if len(df['Exited'].unique()) < 2:
        st.error("The 'Exited' column must contain at least two unique values (0 and 1) for classification.")
        return None, None, None, None, None, None

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ], remainder='passthrough')

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError as e:
        st.warning(f"Could not perform stratified split (e.g., only one class in 'Exited'). Falling back to non-stratified split. Error: {e}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compute class weights for imbalanced datasets
    classes = np.unique(y_train)
    if 0 in classes and 1 in classes:
        class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights_dict = {0: class_weights_array[0], 1: class_weights_array[1]}
        scale_pos_weight_val = class_weights_array[1] / class_weights_array[0]
    else:
        st.warning("Cannot compute class weights: one of the target classes (0 or 1) is missing in the training data. Setting scale_pos_weight to 1.")
        class_weights_dict = {0: 1, 1: 1}
        scale_pos_weight_val = 1

    # --- Random Forest Pipeline ---
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight=class_weights_dict))
    ])
    rf_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10]
    }
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, scoring='roc_auc', n_jobs=1)
    try:
        rf_grid.fit(X_train, y_train)
        rf_best_estimator = rf_grid.best_estimator_
    except Exception as e:
        st.error(f"Random Forest training failed: {e}")
        st.exception(e)
        rf_best_estimator = None

    # --- Neural Network Pipeline ---
    # The preprocessor needs to be fitted once and then used to transform data for NN
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                     loss="binary_crossentropy",
                     metrics=["accuracy"])
    
    # Train NN with class weights
    history = nn_model.fit(X_train_processed, y_train,
                           epochs=50,
                           batch_size=32,
                           validation_data=(X_test_processed, y_test),
                           verbose=0,
                           class_weight=class_weights_dict) # Apply class weights here

    return rf_best_estimator, nn_model, X_test, y_test, X_train, preprocessor


def get_model_predictions(model, X_data, model_type, preprocessor=None):
    """Helper to get predictions and probabilities for a given model."""
    if model_type == 'rf':
        y_pred = model.predict(X_data)
        y_proba = model.predict_proba(X_data)[:, 1]
    elif model_type == 'nn':
        # For NN, X_data needs to be preprocessed first
        X_data_processed = preprocessor.transform(X_data)
        y_proba = model.predict(X_data_processed).ravel()
        y_pred = (y_proba > 0.5).astype(int) # Default threshold for NN
    else:
        raise ValueError("Invalid model_type. Use 'rf' or 'nn'.")
    return y_pred, y_proba


def create_enhanced_metrics_display(y_test, y_pred_rf, y_proba_rf, y_pred_nn, y_proba_nn):
    """Create enhanced metrics display with plotly for both models"""
    metrics_rf = {
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'Recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred_rf, zero_division=0),
        'AUC Score': roc_auc_score(y_test, y_proba_rf)
    }
    metrics_nn = {
        'Accuracy': accuracy_score(y_test, y_pred_nn),
        'Precision': precision_score(y_test, y_pred_nn, zero_division=0),
        'Recall': recall_score(y_test, y_pred_nn, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred_nn, zero_division=0),
        'AUC Score': roc_auc_score(y_test, y_proba_nn)
    }

    # Radar chart for metrics
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=list(metrics_rf.values()),
        theta=list(metrics_rf.keys()),
        fill='toself',
        name='Random Forest Performance',
        line=dict(color='rgb(102, 126, 234)', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=list(metrics_nn.values()),
        theta=list(metrics_nn.keys()),
        fill='toself',
        name='Neural Network Performance',
        line=dict(color='rgb(255, 99, 71)', width=3),
        fillcolor='rgba(255, 99, 71, 0.3)'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickmode='linear', tick0=0, dtick=0.2)),
        showlegend=True,
        title={'text': "Model Performance Metrics Comparison", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        height=400
    )

    return fig_radar, metrics_rf, metrics_nn

def plot_enhanced_confusion_matrix(y_test, y_pred, title_suffix=""):
    """Create enhanced confusion matrix with plotly"""
    cm = confusion_matrix(y_test, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: Stay', 'Predicted: Churn'],
        y=['Actual: Stay', 'Actual: Churn'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20, "color": "white"},
        hoverongaps=False
    ))

    fig.update_layout(
        title={'text': f"Confusion Matrix {title_suffix}", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    return fig

def plot_enhanced_roc_curve(y_test, y_proba_rf, y_proba_nn):
    """Create enhanced ROC curve with plotly for both models"""
    fig = go.Figure()

    # Random Forest ROC
    rf_fpr, rf_tpr, _ = roc_curve(y_test, y_proba_rf)
    rf_auc_score = roc_auc_score(y_test, y_proba_rf)
    fig.add_trace(go.Scatter(
        x=rf_fpr, y=rf_tpr,
        mode='lines',
        name=f'Random Forest (AUC = {rf_auc_score:.3f})',
        line=dict(color='rgb(102, 126, 234)', width=3)
    ))

    # Neural Network ROC
    nn_fpr, nn_tpr, _ = roc_curve(y_test, y_proba_nn)
    nn_auc_score = roc_auc_score(y_test, y_proba_nn)
    fig.add_trace(go.Scatter(
        x=nn_fpr, y=nn_tpr,
        mode='lines',
        name=f'Neural Network (AUC = {nn_auc_score:.3f})',
        line=dict(color='rgb(255, 99, 71)', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))

    fig.update_layout(
        title={'text': "ROC Curves Comparison", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        showlegend=True
    )
    return fig

# Helper function to clean feature names
def clean_feature_name(feature_name):
    if feature_name.startswith('num__'):
        return feature_name.replace('num__', '')
    elif feature_name.startswith('cat__'):
        parts = feature_name.replace('cat__', '').split('_')
        # Handle cases where category might be part of the feature name (e.g., Geography_France)
        if len(parts) > 1 and parts[0] in ['Geography', 'Gender']: # Add more categorical features if needed
            return f"{parts[0].title()}: {parts[1].title()}"
        return feature_name.replace('cat__', '').title()
    else:
        return feature_name.replace('_', ' ').title()

def create_feature_importance_chart(pipeline, X_test, y_test, model_type='rf'):
    """
    Create interactive feature importance charts (built-in and permutation).
    For NN, only permutation importance is directly applicable in this context.
    """
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Get the original feature names (before preprocessing)
        original_feature_names = X_test.columns
        
        # Permutation Importance (applicable to both RF and NN pipelines)
        perm_importance = permutation_importance(
            pipeline, X_test, y_test,
            n_repeats=10, random_state=42, n_jobs=1,
            scoring='roc_auc'
        )
        
        perm_importance_df = pd.DataFrame({
            'feature': original_feature_names,
            'perm_importance': perm_importance.importances_mean,
            'perm_std': perm_importance.importances_std
        })
        perm_importance_df['cleaned_feature'] = perm_importance_df['feature'].apply(clean_feature_name)
        perm_importance_df = perm_importance_df.sort_values('perm_importance', ascending=True)

        fig_perm_importance = go.Figure()
        fig_perm_importance.add_trace(go.Bar(
            x=perm_importance_df['perm_importance'],
            y=perm_importance_df['cleaned_feature'],
            orientation='h',
            error_x=dict(type='data', array=perm_importance_df['perm_std']),
            marker=dict(
                color=perm_importance_df['perm_importance'],
                colorscale='Cividis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{val:.4f}' for val in perm_importance_df['perm_importance']],
            textposition='outside'
        ))
        fig_perm_importance.update_layout(
            title={'text': "Permutation Feature Importance (AUC Score Drop)", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
            xaxis_title="Importance (Mean AUC Drop)",
            yaxis_title="Features",
            height=max(600, len(perm_importance_df) * 30),
            showlegend=False
        )
        
        # Built-in feature importance (only for Random Forest)
        fig_built_in = None
        built_in_importance_df = None
        if model_type == 'rf':
            classifier = pipeline.named_steps['classifier']
            all_feature_names = preprocessor.get_feature_names_out()
            built_in_importance = classifier.feature_importances_
            
            built_in_importance_df = pd.DataFrame({
                'feature': all_feature_names,
                'importance': built_in_importance
            })
            built_in_importance_df['cleaned_feature'] = built_in_importance_df['feature'].apply(clean_feature_name)
            built_in_importance_df = built_in_importance_df.sort_values('importance', ascending=True)

            fig_built_in = go.Figure()
            fig_built_in.add_trace(go.Bar(
                x=built_in_importance_df['importance'],
                y=built_in_importance_df['cleaned_feature'],
                orientation='h',
                marker=dict(
                    color=built_in_importance_df['importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f'{val:.4f}' for val in built_in_importance_df['importance']],
                textposition='outside'
            ))
            fig_built_in.update_layout(
                title={'text': "Built-in Feature Importance (Random Forest)", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
                xaxis_title="Feature Importance Score",
                yaxis_title="Features",
                height=max(600, len(built_in_importance_df) * 30),
                showlegend=False
            )

        return fig_built_in, built_in_importance_df, fig_perm_importance, perm_importance_df
            
    except Exception as e:
        st.error(f"Error creating feature importance chart: {e}")
        st.exception(e)
        return None, None, None, None

def create_data_explorer_charts(df):
    """Create interactive charts for data exploration"""
    charts = {}

    # Age distribution
    fig_age = px.histogram(
        df, x='Age', color='Exited',
        title='Age Distribution by Churn Status',
        nbins=20,
        color_discrete_map={0: '#48dbfb', 1: '#ff6b6b'},
        hover_data={'Age': ':.0f', 'Exited': True}
    )
    fig_age.update_layout(height=400, showlegend=True)
    charts['age'] = fig_age

    # Balance distribution
    fig_balance = px.box(
        df, x='Exited', y='Balance',
        title='Balance Distribution by Churn Status',
        color='Exited',
        color_discrete_map={0: '#48dbfb', 1: '#ff6b6b'},
        hover_data={'Balance': ':.2f'}
    )
    fig_balance.update_layout(height=400, showlegend=True)
    charts['balance'] = fig_balance

    # Geography distribution
    geography_counts = df.groupby(['Geography', 'Exited']).size().reset_index(name='count')
    fig_geo = px.bar(
        geography_counts, x='Geography', y='count', color='Exited',
        title='Churn Distribution by Geography',
        color_discrete_map={0: '#48dbfb', 1: '#ff6b6b'},
        barmode='group',
        text_auto=True
    )
    fig_geo.update_layout(height=400, showlegend=True)
    charts['geography'] = fig_geo

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if 'Exited' in numeric_df.columns:
        correlation_matrix = numeric_df.corr()
    else:
        correlation_matrix = numeric_df.corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig_corr.update_layout(
        title='Feature Correlation Matrix',
        height=500
    )
    charts['correlation'] = fig_corr

    return charts

def analyze_customer_prediction(rf_pipeline, nn_model, customer_data, X_train_original_columns, preprocessor):
    """Analyze why a customer was predicted to churn or not for both models"""
    try:
        # Ensure customer_data has all columns present in X_train for the preprocessor
        customer_data_aligned = pd.DataFrame(columns=X_train_original_columns)
        for col in X_train_original_columns:
            if col in customer_data.columns:
                customer_data_aligned[col] = customer_data[col]
            else:
                # Fill missing columns: numerical to 0, categorical to 'Unknown'
                # Get feature names from preprocessor for robust type inference
                num_features_in_preprocessor = [name for name, _, features in preprocessor.transformers_ if name == 'num'][0][2] if 'num' in [t[0] for t in preprocessor.transformers_] else []
                cat_features_in_preprocessor = [name for name, _, features in preprocessor.transformers_ if name == 'cat'][0][2] if 'cat' in [t[0] for t in preprocessor.transformers_] else []

                if col in num_features_in_preprocessor:
                    customer_data_aligned[col] = 0.0
                elif col in cat_features_in_preprocessor:
                    customer_data_aligned[col] = 'Unknown'
                else:
                    customer_data_aligned[col] = np.nan

        customer_data_aligned = customer_data_aligned.fillna(0)
        customer_data_for_prediction = customer_data_aligned.iloc[0:1]
        
        # --- Random Forest Prediction ---
        rf_pred_proba = rf_pipeline.predict_proba(customer_data_for_prediction)[0, 1]
        rf_pred_class = rf_pipeline.predict(customer_data_for_prediction)[0]

        # --- Neural Network Prediction ---
        nn_pred_proba = nn_model.predict(preprocessor.transform(customer_data_for_prediction)).ravel()[0]
        nn_pred_class = (nn_pred_proba > 0.5).astype(int)

        # Feature importance for Random Forest (built-in)
        rf_model = rf_pipeline.named_steps['classifier']
        rf_feature_importance = rf_model.feature_importances_
        preprocessor_feature_names = preprocessor.get_feature_names_out()

        analysis_df = pd.DataFrame({
            'feature': preprocessor_feature_names,
            'importance': rf_feature_importance
        })
        analysis_df['cleaned_feature'] = analysis_df['feature'].apply(clean_feature_name)
        analysis_df = analysis_df.sort_values('importance', ascending=False).reset_index(drop=True)

        return rf_pred_proba, rf_pred_class, nn_pred_proba, nn_pred_class, analysis_df

    except Exception as e:
        st.error(f"Error analyzing customer prediction: {e}")
        st.exception(e)
        return None, None, None, None, None

def create_individual_analysis_chart(analysis_df):
    """Create individual customer analysis chart"""
    top_features = analysis_df.head(10).copy()
    
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_features['importance'],
        y=top_features['cleaned_feature'],
        orientation='h',
        marker=dict(
            color=top_features['importance'],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=[f'{val:.4f}' for val in top_features['importance']],
        textposition='outside'
    ))

    fig.update_layout(
        title={'text': "Top 10 Feature Contributions (Random Forest)", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
        xaxis_title="Feature Importance",
        yaxis_title="Features",
        height=500,
        showlegend=False
    )
    return fig

# --- Main App ---

# Enhanced header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🚀 Advanced AI Customer Churn Prediction</h1>
    <p class="header-subtitle">
        Harness the power of Random Forest and Neural Networks to predict customer churn with precision.<br>
        Empowering businesses to retain customers through predictive analytics
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'rf_pipeline' not in st.session_state:
    st.session_state.rf_pipeline = None
if 'nn_model' not in st.session_state:
    st.session_state.nn_model = None
if 'X_train_cols' not in st.session_state:
    st.session_state.X_train_cols = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# Enhanced sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration Center")
    
    # Optional File uploader with enhanced styling
    uploaded_file = st.file_uploader(
        "📁 Upload Your Own Data (Optional)",
        type=['csv'],
        help="Upload a CSV file to override the default data. Must contain: CreditScore, Geography, Gender, Age, Tenure, Balance, EstimatedSalary, Exited"
    )
    
    # Load data using the updated function
    df = load_data(uploaded_file=uploaded_file)
    
    # Model information section
    st.markdown("---")
    st.markdown("### 🤖 Model Information")
    
    # Retrain button with enhanced styling
    if st.button("🔄 Retrain Models", use_container_width=True):
        st.session_state.models_trained = False
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

if df is not None and not df.empty:
    # Enhanced dataset overview
    st.markdown("### 📊 Dataset Overview")
    
    # Create enhanced metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        churned = df['Exited'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{churned:,}</div>
            <div class="metric-label">Churned Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        churn_rate = df['Exited'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{churn_rate:.1%}</div>
            <div class="metric-label">Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        balance_counts = df['Exited'].value_counts()
        if 0 in balance_counts and 1 in balance_counts:
            ratio_text = f"{balance_counts[0]}:{balance_counts[1]}"
        else:
            ratio_text = "N/A (single class)"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{ratio_text}</div>
            <div class="metric-label">Class Balance (Stay:Churn)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Train models if not already trained
    if not st.session_state.models_trained or st.session_state.rf_pipeline is None or st.session_state.nn_model is None:
        with st.spinner("🚀 Training Random Forest and Neural Network models..."):
            progress_bar = st.progress(0)
            import time
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            rf_pipeline, nn_model, X_test, y_test, X_train, preprocessor = train_models(df)
            
        if rf_pipeline and nn_model:
            st.success("✅ Models trained successfully!")
            st.session_state.rf_pipeline = rf_pipeline
            st.session_state.nn_model = nn_model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_train_cols = X_train.columns.tolist()
            st.session_state.preprocessor = preprocessor
            st.session_state.models_trained = True
        else:
            st.error("❌ Model training failed. Please check your data and try again.")
    
    # Access trained models from session state
    rf_pipeline = st.session_state.get('rf_pipeline')
    nn_model = st.session_state.get('nn_model')
    X_test = st.session_state.get('X_test')
    y_test = st.session_state.get('y_test')
    X_train_original_columns = st.session_state.get('X_train_cols')
    preprocessor = st.session_state.get('preprocessor')
    
    if rf_pipeline and nn_model and X_test is not None and y_test is not None and X_train_original_columns is not None and preprocessor is not None:
        # Display model info in sidebar
        with st.sidebar:
            st.markdown("### ✨ Model Summary")
            st.markdown(f"""
            - **Random Forest Estimators**: {rf_pipeline.named_steps['classifier'].n_estimators}
            - **Random Forest Max Depth**: {rf_pipeline.named_steps['classifier'].max_depth}
            - **Neural Network Layers**: {len(nn_model.layers)}
            """)
            st.success("🎯 Models ready for predictions!")
        
        # Enhanced tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Data Explorer",
            "📈 Model Performance",
            "🎯 Live Prediction",
            "📊 Batch Prediction"
        ])
        
        # Tab 1: Data Explorer
        with tab1:
            st.markdown("### 🔍 Data Explorer")
            
            # Create data exploration charts
            charts = create_data_explorer_charts(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(charts['age'], use_container_width=True)
                st.plotly_chart(charts['geography'], use_container_width=True)
            
            with col2:
                st.plotly_chart(charts['balance'], use_container_width=True)
                st.plotly_chart(charts['correlation'], use_container_width=True)
            
            # Enhanced data table
            st.markdown("### 📋 Raw Data Sample")
            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Data statistics
            st.markdown("### 📊 Statistical Summary")
            st.dataframe(
                df.describe(),
                use_container_width=True
            )
        
        # Tab 2: Model Performance
        with tab2:
            st.markdown("### 📈 Model Performance Analytics")
            
            # Generate predictions for test set for both models
            y_pred_rf, y_proba_rf = get_model_predictions(rf_pipeline, X_test, 'rf')
            y_pred_nn, y_proba_nn = get_model_predictions(nn_model, X_test, 'nn', preprocessor)
            
            # Create enhanced metrics display
            metrics_fig, metrics_dict_rf, metrics_dict_nn = create_enhanced_metrics_display(y_test, y_pred_rf, y_proba_rf, y_pred_nn, y_proba_nn)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(metrics_fig, use_container_width=True)
                
                # Random Forest Confusion Matrix
                cm_fig_rf = plot_enhanced_confusion_matrix(y_test, y_pred_rf, "(Random Forest)")
                st.plotly_chart(cm_fig_rf, use_container_width=True)

                # Neural Network Confusion Matrix
                cm_fig_nn = plot_enhanced_confusion_matrix(y_test, y_pred_nn, "(Neural Network)")
                st.plotly_chart(cm_fig_nn, use_container_width=True)
            
            with col2:
                # Enhanced ROC curve for both models
                roc_fig = plot_enhanced_roc_curve(y_test, y_proba_rf, y_proba_nn)
                st.plotly_chart(roc_fig, use_container_width=True)
                
                # Metrics summary
                st.markdown("### 📊 Performance Metrics")
                st.subheader("Random Forest Metrics:")
                for metric, value in metrics_dict_rf.items():
                    st.metric(metric, f"{value:.3f}")
                
                st.subheader("Neural Network Metrics:")
                for metric, value in metrics_dict_nn.items():
                    st.metric(metric, f"{value:.3f}")
            
            # Feature importance analysis
            st.markdown("### 🎯 Feature Importance Analysis")

            # Random Forest Built-in Importance
            rf_built_in_fig, rf_built_in_df, perm_importance_fig, perm_importance_df = create_feature_importance_chart(rf_pipeline, X_test, y_test, model_type='rf')

            if rf_built_in_fig is not None:
                st.plotly_chart(rf_built_in_fig, use_container_width=True)
            else:
                st.info("Built-in feature importance is primarily available for tree-based models like Random Forest.")
            
            # Permutation Importance (applicable to both models via the pipeline)
            if perm_importance_fig is not None:
                st.plotly_chart(perm_importance_fig, use_container_width=True)
            else:
                st.error("Could not generate permutation feature importance analysis.")
            
            # Top features summary (using permutation importance for a general view)
            st.markdown("### 🔝 Top 5 Most Important Features (Permutation Importance)")
            if perm_importance_df is not None:
                top_features = perm_importance_df.sort_values('perm_importance', ascending=False).head(5)
                for idx, row in top_features.iterrows():
                    st.write(f"**{row['cleaned_feature']}**: {row['perm_importance']:.4f} ± {row['perm_std']:.4f}")
            else:
                st.error("Could not retrieve top features from permutation importance.")
        
        # Tab 3: Live Prediction
        with tab3:
            st.markdown("### 🎯 Individual Customer Prediction")
            
            # Enhanced prediction form
            st.markdown('<div class="prediction-form">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💳 Financial Information")
                credit_score = st.slider("Credit Score", 300, 850, 650)
                balance = st.number_input("Account Balance ($)", 0.0, 300000.0, 50000.0)
                estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 75000.0)
            
            
            with col2:
                st.markdown("#### 👤 Personal Information")
                age = st.slider("Age", 18, 80, 40)
                tenure = st.slider("Tenure (years)", 0, 10, 3)
                geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
                gender = st.selectbox("Gender", ['Male', 'Female'])
            
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔮 Predict Churn Risk", use_container_width=True):
                # Create customer data
                customer_data = pd.DataFrame({
                    'CreditScore': [credit_score],
                    'Geography': [geography],
                    'Gender': [gender],
                    'Age': [age],
                    'Tenure': [tenure],
                    'Balance': [balance],
                    'EstimatedSalary': [estimated_salary]
                })
                
                # Make prediction
                rf_pred_proba, rf_pred_class, nn_pred_proba, nn_pred_class, analysis_df = analyze_customer_prediction(
                    rf_pipeline, nn_model, customer_data, X_train_original_columns, preprocessor
                )
                
                if rf_pred_proba is not None:
                    st.markdown("### 🎯 Prediction Results")
                    col_rf, col_nn = st.columns(2)

                    with col_rf:
                        st.subheader("Random Forest Prediction")
                        rf_risk_level = "High Risk" if rf_pred_proba > 0.7 else "Medium Risk" if rf_pred_proba > 0.3 else "Low Risk"
                        rf_risk_color = "prediction-high-risk" if rf_pred_proba > 0.7 else "prediction-medium-risk" if rf_pred_proba > 0.3 else "prediction-low-risk"
                        st.markdown(f"""
                        <div class="prediction-result {rf_risk_color}">
                            <h3>Churn Probability: {rf_pred_proba:.1%}</h3>
                            <h3>Risk Level: {rf_risk_level}</h3>
                            <p>{"⚠️ This customer is likely to churn. Consider retention strategies." if rf_pred_class == 1 else "✅ This customer is likely to stay. Continue providing excellent service."}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_nn:
                        st.subheader("Neural Network Prediction")
                        nn_risk_level = "High Risk" if nn_pred_proba > 0.7 else "Medium Risk" if nn_pred_proba > 0.3 else "Low Risk"
                        nn_risk_color = "prediction-high-risk" if nn_pred_proba > 0.7 else "prediction-medium-risk" if nn_pred_proba > 0.3 else "prediction-low-risk"
                        st.markdown(f"""
                        <div class="prediction-result {nn_risk_color}">
                            <h3>Churn Probability: {nn_pred_proba:.1%}</h3>
                            <h3>Risk Level: {nn_risk_level}</h3>
                            <p>{"⚠️ This customer is likely to churn. Consider retention strategies." if nn_pred_class == 1 else "✅ This customer is likely to stay. Continue providing excellent service."}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Individual analysis chart (using Random Forest importance for interpretability)
                    if analysis_df is not None:
                        st.markdown("### 🔍 Prediction Analysis (Random Forest Feature Contributions)")
                        individual_chart = create_individual_analysis_chart(analysis_df)
                        st.plotly_chart(individual_chart, use_container_width=True)
                        
                        # Detailed feature analysis
                        st.markdown("### 📊 Feature Contribution Details")
                        st.dataframe(
                            analysis_df.head(10)[['cleaned_feature', 'importance']].rename(columns={'cleaned_feature': 'Feature', 'importance': 'Importance'}),
                            use_container_width=True,
                            hide_index=True
                        )
        
        # Tab 4: Batch Prediction
        with tab4:
            st.markdown("### 📊 Batch Prediction")
            
            # Batch prediction file uploader
            batch_file = st.file_uploader(
                "📁 Upload CSV for Batch Prediction",
                type=['csv'],
                help="Upload a CSV file with customer data for batch prediction. It should contain the same columns as the training data, excluding the 'Exited' column."
            )
            
            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)
                    st.success(f"✅ Loaded {len(batch_df)} customers for prediction")
                    
                    # Ensure the batch_df only contains features expected by the pipeline
                    if 'Exited' in batch_df.columns:
                        st.warning("⚠️ 'Exited' column found in uploaded batch file. It will be ignored for prediction.")
                        batch_df_for_prediction = batch_df.drop('Exited', axis=1)
                    else:
                        batch_df_for_prediction = batch_df.copy()
                    
                    # Drop RowNumber, CustomerId, Surname if they exist in the batch file
                    cols_to_drop_batch = ['RowNumber', 'CustomerId', 'Surname']
                    for col in cols_to_drop_batch:
                        if col in batch_df_for_prediction.columns:
                            batch_df_for_prediction = batch_df_for_prediction.drop(columns=[col])

                    missing_cols = set(X_train_original_columns) - set(batch_df_for_prediction.columns)
                    extra_cols = set(batch_df_for_prediction.columns) - set(X_train_original_columns)
                    
                    if missing_cols:
                        st.warning(f"Batch file is missing columns expected by the model: {', '.join(missing_cols)}. These will be treated as absent (e.g., 0 for numerical, new category for categorical).")
                        for col in missing_cols:
                            # Infer type from the preprocessor's transformers
                            num_features_in_preprocessor = [name for name, _, features in preprocessor.transformers_ if name == 'num'][0][2] if 'num' in [t[0] for t in preprocessor.transformers_] else []
                            cat_features_in_preprocessor = [name for name, _, features in preprocessor.transformers_ if name == 'cat'][0][2] if 'cat' in [t[0] for t in preprocessor.transformers_] else []

                            if col in num_features_in_preprocessor:
                                batch_df_for_prediction[col] = 0.0
                            elif col in cat_features_in_preprocessor:
                                batch_df_for_prediction[col] = 'Unknown'
                            else:
                                batch_df_for_prediction[col] = np.nan # Fallback
                    
                    if extra_cols:
                        st.warning(f"Batch file contains extra columns not used by the model: {', '.join(extra_cols)}. These will be ignored.")
                        batch_df_for_prediction = batch_df_for_prediction[[col for col in X_train_original_columns if col in batch_df_for_prediction.columns]]
                    
                    batch_df_for_prediction = batch_df_for_prediction.reindex(columns=X_train_original_columns, fill_value=0)
                    
                    # Display sample data used for prediction
                    st.markdown("### 📋 Sample Data (for prediction)")
                    st.dataframe(batch_df_for_prediction.head(), use_container_width=True)
                    
                    if st.button("🚀 Generate Batch Predictions", use_container_width=True):
                        with st.spinner("🔄 Generating predictions..."):
                            # RF Predictions
                            rf_proba_output = rf_pipeline.predict_proba(batch_df_for_prediction)
                            rf_batch_predictions = rf_proba_output[:, 1]
                            rf_batch_classes = rf_pipeline.predict(batch_df_for_prediction)

                            # NN Predictions
                            nn_proba_output = nn_model.predict(preprocessor.transform(batch_df_for_prediction)).ravel()
                            nn_batch_predictions = nn_proba_output
                            nn_batch_classes = (nn_proba_output > 0.5).astype(int)
                            
                            batch_df['RF_Churn_Probability'] = rf_batch_predictions
                            batch_df['RF_Churn_Prediction'] = rf_batch_classes
                            batch_df['RF_Risk_Level'] = batch_df['RF_Churn_Probability'].apply(
                                lambda x: 'High Risk' if x > 0.7 else 'Medium Risk' if x > 0.3 else 'Low Risk'
                            )

                            batch_df['NN_Churn_Probability'] = nn_batch_predictions
                            batch_df['NN_Churn_Prediction'] = nn_batch_classes
                            batch_df['NN_Risk_Level'] = batch_df['NN_Churn_Probability'].apply(
                                lambda x: 'High Risk' if x > 0.7 else 'Medium Risk' if x > 0.3 else 'Low Risk'
                            )
                            
                            st.markdown("### 🎯 Batch Prediction Results")
                            st.dataframe(batch_df, use_container_width=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Random Forest Risk Distribution")
                                rf_risk_counts = batch_df['RF_Risk_Level'].value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk'])
                                rf_risk_counts = rf_risk_counts.fillna(0)
                                fig_rf_risk = px.pie(
                                    values=rf_risk_counts.values, names=rf_risk_counts.index, title='RF Risk Level Distribution',
                                    color_discrete_map={'High Risk': '#ff6b6b', 'Medium Risk': '#feca57', 'Low Risk': '#48dbfb'}
                                )
                                st.plotly_chart(fig_rf_risk, use_container_width=True)
                            with col2:
                                st.subheader("Neural Network Risk Distribution")
                                nn_risk_counts = batch_df['NN_Risk_Level'].value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk'])
                                nn_risk_counts = nn_risk_counts.fillna(0)
                                fig_nn_risk = px.pie(
                                    values=nn_risk_counts.values, names=nn_risk_counts.index, title='NN Risk Level Distribution',
                                    color_discrete_map={'High Risk': '#ff6b6b', 'Medium Risk': '#feca57', 'Low Risk': '#48dbfb'}
                                )
                                st.plotly_chart(fig_nn_risk, use_container_width=True)
                            
                            csv = batch_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download All Predictions",
                                data=csv,
                                file_name='batch_predictions_combined.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                            
                except Exception as e:
                    st.error(f"❌ Error processing batch file: {e}. Please ensure your CSV has the correct format and data types.")
                    st.exception(e)
            
            else:
                st.info("📁 Please upload a CSV file for batch prediction")
                
                st.markdown("### 📝 Required CSV Format")
                sample_format = pd.DataFrame({
                    'CreditScore': [650, 720, 580],
                    'Geography': ['France', 'Spain', 'Germany'],
                    'Gender': ['Male', 'Female', 'Male'],
                    'Age': [35, 45, 28],
                    'Tenure': [3, 7, 1],
                    'Balance': [50000, 75000, 0],
                    'EstimatedSalary': [75000, 85000, 60000]
                })
                
                st.dataframe(sample_format, use_container_width=True)
                
                csv_sample = sample_format.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Sample Format",
                    data=csv_sample,
                    file_name='sample_format.csv',
                    mime='text/csv',
                    use_container_width=True
                )
    
    else:
        st.error("❌ Models not available. Please ensure data is loaded and models are trained successfully.")

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-top: 2rem;">
    <h3>🚀 Advanced AI Customer Churn Prediction</h3>
    <p>Combining the power of Random Forest and Neural Networks with interactive analytics</p>
    <p>Empowering businesses to retain customers through predictive analytics</p>
</div>
""", unsafe_allow_html=True)
