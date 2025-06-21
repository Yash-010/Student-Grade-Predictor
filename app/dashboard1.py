import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTab [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTab [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and preprocess the uploaded data"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

def preprocess_data(df):
    """Preprocess the data for modeling"""
    df_clean = df.copy()
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # Feature engineering
    if all(col in df_clean.columns for col in ['G1', 'G2']):
        df_clean['grade_trend'] = df_clean['G2'] - df_clean['G1']
        df_clean['avg_previous_grades'] = (df_clean['G1'] + df_clean['G2']) / 2
    
    if all(col in df_clean.columns for col in ['Medu', 'Fedu']):
        df_clean['parent_edu_sum'] = df_clean['Medu'] + df_clean['Fedu']
    
    if all(col in df_clean.columns for col in ['Dalc', 'Walc']):
        df_clean['total_alcohol'] = df_clean['Dalc'] + df_clean['Walc']
    
    # Encode categorical variables
    encoders = {}
    for col in df_clean.select_dtypes(include=['object']).columns:
        if col != 'G3':  # Don't encode target if it's categorical
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            encoders[col] = le
    
    return df_clean, encoders

@st.cache_resource
def train_models(X_train, y_train):
    """Train multiple models and return them"""
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and return results"""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R²': r2_score(y_test, y_pred)
        }
    return results

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Student Grade Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload Student Dataset", 
        type=['csv'],
        help="Upload a CSV file with student data including G3 as target variable"
    )
    
    if uploaded_file is None:
        st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome to Student Grade Predictor!</h3>
            <p>This dashboard helps you:</p>
            <ul>
                <li>📈 Analyze student performance data</li>
                <li>🤖 Build ML models to predict grades</li>
                <li>📊 Visualize insights and patterns</li>
                <li>🎯 Make predictions for new students</li>
            </ul>
            <p><strong>To get started:</strong> Upload your CSV file using the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        sample_data = {
            'school': ['GP', 'MS', 'GP'],
            'sex': ['F', 'M', 'F'],
            'age': [18, 17, 15],
            'address': ['U', 'U', 'R'],
            'studytime': [2, 2, 1],
            'failures': [0, 0, 3],
            'G1': [5, 6, 10],
            'G2': [6, 5, 11],
            'G3': [6, 6, 11]
        }
        st.dataframe(pd.DataFrame(sample_data))
        return
    
    # Load data
    df = load_and_process_data(uploaded_file)
    
    if df is not None:
        # Data preprocessing
        df_clean, encoders = preprocess_data(df)
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "🔍 EDA", 
            "🤖 Model Training", 
            "📈 Results", 
            "🎯 Prediction"
        ])
        
        with tab1:
            show_data_overview(df, df_clean)
        
        with tab2:
            show_eda(df_clean)
        
        with tab3:
            show_model_training(df_clean)
        
        with tab4:
            show_results(df_clean)
        
        with tab5:
            show_prediction_interface(df_clean, encoders)

def show_data_overview(df, df_clean):
    """Display data overview section"""
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Total Samples</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Features</p>
        </div>
        """.format(df.shape[1]-1), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Missing Values</p>
        </div>
        """.format(df.isnull().sum().sum()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>{:.1f}</h3>
            <p>Avg Grade (G3)</p>
        </div>
        """.format(df['G3'].mean() if 'G3' in df.columns else 0), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📈 Data Info")
        
        # Data types
        st.write("**Data Types:**")
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Type': df.dtypes.values
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write("**Missing Values:**")
            missing_df = missing_data[missing_data > 0].reset_index()
            missing_df.columns = ['Column', 'Missing']
            st.dataframe(missing_df, use_container_width=True)

def show_eda(df_clean):
    """Display EDA section"""
    st.header("🔍 Exploratory Data Analysis")
    
    if 'G3' not in df_clean.columns:
        st.error("G3 column not found in the dataset!")
        return
    
    # Target variable analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Target Variable Distribution")
        fig = px.histogram(df_clean, x='G3', nbins=20, 
                          title="Distribution of Final Grades (G3)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Grade Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Value': [
                df_clean['G3'].mean(),
                df_clean['G3'].median(),
                df_clean['G3'].std(),
                df_clean['G3'].min(),
                df_clean['G3'].max()
            ]
        }).round(2)
        st.dataframe(stats_df, use_container_width=True)
        
        # Grade categories
        st.write("**Grade Categories:**")
        grade_bins = pd.cut(df_clean['G3'], bins=[0, 10, 15, 20], labels=['Low', 'Medium', 'High'])
        grade_dist = grade_bins.value_counts()
        for grade, count in grade_dist.items():
            st.write(f"• {grade}: {count} students ({count/len(df_clean)*100:.1f}%)")
    
    st.markdown("---")
    
    # Correlation analysis
    st.subheader("🔗 Feature Correlations")
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df_clean[numeric_cols].corr()
        
        # Top correlations with G3
        if 'G3' in corr_matrix.columns:
            g3_corr = corr_matrix['G3'].abs().sort_values(ascending=False)[1:11]  # Top 10
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = px.bar(x=g3_corr.values, y=g3_corr.index, 
                           title="Top 10 Features Correlated with G3",
                           labels={'x': 'Absolute Correlation', 'y': 'Features'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Heatmap of top correlated features
                top_features = ['G3'] + g3_corr.head(8).index.tolist()
                corr_subset = corr_matrix.loc[top_features, top_features]
                
                fig = px.imshow(corr_subset, text_auto=True, aspect="auto",
                              title="Correlation Heatmap - Top Features")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
    
    # Grade progression analysis
    if all(col in df_clean.columns for col in ['G1', 'G2', 'G3']):
        st.markdown("---")
        st.subheader("📈 Grade Progression Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df_clean, x='G1', y='G3', opacity=0.6,
                           title="First Period vs Final Grade")
            fig.add_shape(type="line", x0=0, y0=0, x1=20, y1=20,
                         line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df_clean, x='G2', y='G3', opacity=0.6,
                           title="Second Period vs Final Grade")
            fig.add_shape(type="line", x0=0, y0=0, x1=20, y1=20,
                         line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)

def show_model_training(df_clean):
    """Display model training section"""
    st.header("🤖 Model Training & Evaluation")
    
    if 'G3' not in df_clean.columns:
        st.error("G3 column not found!")
        return
    
    # Prepare data
    X = df_clean.drop('G3', axis=1)
    y = df_clean['G3']
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models... Please wait"):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train models
            models = train_models(X_train, y_train)
            
            # Evaluate models
            results = evaluate_models(models, X_test, y_test)
            
            # Store in session state
            st.session_state['models'] = models
            st.session_state['results'] = results
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = X.columns.tolist()
        
        st.success("✅ Models trained successfully!")
    
    # Show training parameters
    st.subheader("⚙️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Samples", len(df_clean))
    with col2:
        st.metric("🎯 Features", len(X.columns))
    with col3:
        st.metric("✂️ Train/Test Split", "80/20")
    
    # Model details
    st.subheader("🔧 Model Details")
    
    model_info = {
        'Random Forest': "Ensemble method using multiple decision trees",
        'Gradient Boosting': "Sequential ensemble building strong learners",
        'Linear Regression': "Simple linear relationship modeling",
        'Ridge Regression': "Linear regression with L2 regularization"
    }
    
    for model, description in model_info.items():
        st.write(f"**{model}:** {description}")

def show_results(df_clean):
    """Display results section"""
    st.header("📈 Model Results & Performance")
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Please train models first in the 'Model Training' tab!")
        return
    
    results = st.session_state['results']
    models = st.session_state['models']
    
    # Performance metrics table
    st.subheader("📊 Performance Comparison")
    
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    results_df = results_df.sort_values('R²', ascending=False)
    
    # Style the dataframe
    st.dataframe(
        results_df.style.highlight_max(axis=0, subset=['R²'])
                        .highlight_min(axis=0, subset=['RMSE', 'MAE']),
        use_container_width=True
    )
    
    # Best model highlight
    best_model = results_df.index[0]
    st.success(f"🏆 **Best Model:** {best_model} (R² = {results_df.loc[best_model, 'R²']:.4f})")
    
    # Visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(x=results_df.index, y=results_df['R²'],
                    title="R² Score Comparison",
                    labels={'x': 'Models', 'y': 'R² Score'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(x=results_df.index, y=results_df['RMSE'],
                    title="RMSE Comparison",
                    labels={'x': 'Models', 'y': 'RMSE'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction vs Actual plot for best model
    st.subheader(f"🎯 {best_model} - Prediction Analysis")
    
    best_model_obj = models[best_model]
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    y_pred = best_model_obj.predict(X_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(x=y_test, y=y_pred, opacity=0.6,
                        title="Actual vs Predicted Values")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), 
                     x1=y_test.max(), y1=y_test.max(),
                     line=dict(color="red", dash="dash"))
        fig.update_layout(
            xaxis_title="Actual G3",
            yaxis_title="Predicted G3",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Residuals plot
        residuals = y_test - y_pred
        fig = px.scatter(x=y_pred, y=residuals, opacity=0.6,
                        title="Residuals Plot")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            xaxis_title="Predicted G3",
            yaxis_title="Residuals",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (for tree-based models)
    if hasattr(best_model_obj, 'feature_importances_'):
        st.subheader("🔍 Feature Importance")
        
        feature_names = st.session_state['feature_names']
        importances = best_model_obj.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(importance_df, x='Importance', y='Feature',
                    title="Top 15 Feature Importances",
                    orientation='h')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_interface(df_clean, encoders):
    """Display prediction interface"""
    st.header("🎯 Make Predictions")
    
    if 'models' not in st.session_state:
        st.warning("⚠️ Please train models first in the 'Model Training' tab!")
        return
    
    models = st.session_state['models']
    best_model_name = max(st.session_state['results'].items(), 
                         key=lambda x: x[1]['R²'])[0]
    
    st.subheader(f"Using Best Model: {best_model_name}")
    
    # Input form for prediction
    st.markdown("### 📝 Enter Student Information")
    
    # Create input fields based on the dataset
    input_data = {}
    
    # Get sample data structure
    sample_student = df_clean.drop('G3', axis=1).iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📚 Academic Info**")
        if 'school' in df_clean.columns:
            input_data['school'] = st.selectbox("School", options=[0, 1], 
                                              help="0: GP, 1: MS")
        if 'studytime' in df_clean.columns:
            input_data['studytime'] = st.slider("Study Time", 1, 4, 2)
        if 'failures' in df_clean.columns:
            input_data['failures'] = st.slider("Past Failures", 0, 4, 0)
        if 'G1' in df_clean.columns:
            input_data['G1'] = st.slider("First Period Grade", 0, 20, 10)
        if 'G2' in df_clean.columns:
            input_data['G2'] = st.slider("Second Period Grade", 0, 20, 10)
    
    with col2:
        st.markdown("**👤 Personal Info**")
        if 'sex' in df_clean.columns:
            input_data['sex'] = st.selectbox("Gender", options=[0, 1], 
                                           help="0: Female, 1: Male")
        if 'age' in df_clean.columns:
            input_data['age'] = st.slider("Age", 15, 22, 17)
        if 'address' in df_clean.columns:
            input_data['address'] = st.selectbox("Address", options=[0, 1], 
                                               help="0: Rural, 1: Urban")
        if 'absences' in df_clean.columns:
            input_data['absences'] = st.slider("Absences", 0, 50, 5)
    
    with col3:
        st.markdown("**👨‍👩‍👧‍👦 Family Info**")
        if 'Medu' in df_clean.columns:
            input_data['Medu'] = st.slider("Mother's Education", 0, 4, 2)
        if 'Fedu' in df_clean.columns:
            input_data['Fedu'] = st.slider("Father's Education", 0, 4, 2)
        if 'famrel' in df_clean.columns:
            input_data['famrel'] = st.slider("Family Relationship", 1, 5, 3)
        if 'freetime' in df_clean.columns:
            input_data['freetime'] = st.slider("Free Time", 1, 5, 3)
    
    # Fill remaining features with median values
    feature_names = st.session_state['feature_names']
    for feature in feature_names:
        if feature not in input_data:
            input_data[feature] = df_clean[feature].median()
    
    # Prediction button
    if st.button("🔮 Predict Grade", type="primary"):
        # Create input dataframe
        input_df = pd.DataFrame([input_data])
        
        expected_features = st.session_state['feature_names']
        input_df = input_df[expected_features] 

        # # Make prediction
        best_model = models[best_model_name]
        prediction = best_model.predict(input_df)[0]
        
        # Display result
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🎯 Prediction Result")
            
            # Determine grade category and color
            if prediction >= 16:
                grade_category = "Excellent"
                color = "green"
            elif prediction >= 14:
                grade_category = "Good"
                color = "blue"
            elif prediction >= 10:
                grade_category = "Average"
                color = "orange"
            else:
                grade_category = "Needs Improvement"
                color = "red"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 10px; color: white; margin: 1rem 0;">
                <h2>Predicted Final Grade (G3)</h2>
                <h1 style="font-size: 4rem; margin: 0;">{prediction:.1f}</h1>
                <h3 style="color: {color};">{grade_category}</h3>
            </div>
            """, unsafe_allow_html=True)
            
        # Show confidence interval
        st.info(f"📊 **Model Confidence:** R² = {st.session_state['results'][best_model_name]['R²']:.3f}")
        
        # Recommendations based on prediction
        st.markdown("### 💡 Recommendations")
        
        if prediction < 10:
            st.error("""
            🚨 **High Risk Student** - Immediate intervention needed:
            - Increase study time and provide additional support
            - Reduce absences and improve attendance
            - Consider tutoring or extra help sessions
            """)
        elif prediction < 14:
            st.warning("""
            ⚠️ **Moderate Risk** - Additional support recommended:
            - Monitor progress closely
            - Encourage consistent study habits
            - Provide motivational support
            """)
        else:
            st.success("""
            ✅ **Good Performance** - Continue current approach:
            - Maintain current study habits
            - Consider advanced coursework
            - Peer tutoring opportunities
            """)

if __name__ == "__main__":
    main()