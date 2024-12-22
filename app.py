import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #FF4B4B;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    h1 {
        color: #FF4B4B;
    }
    h2 {
        color: #FF4B4B;
    }
    h3 {
        color: #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=100)
    st.title("Navigation")
    page = st.radio("", ["🏠 Home", "📊 Analysis", "🔮 Prediction"])
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This application helps predict diabetes using machine learning. "
        "It analyzes various health parameters to provide insights and predictions."
    )

# Load models and data
@st.cache_resource
def load_models():
    try:
        scaler = pickle.load(open("Model/Standarscaler.pkl", "rb"))
        model = pickle.load(open("Model/Classifier.pkl", "rb"))
        return scaler, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Dataset/diabetes_cleaned1.csv")
        df1 = pd.read_csv("Dataset/diabetes_cleaned2.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y, df, df1
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

Standard_scaler, logistic_model = load_models()
X, y, df, df1 = load_data()

# Home Page
if page == "🏠 Home":
    st.title("🏥 Diabetes Prediction System")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Model Accuracy", "82.17%")
    with col3:
        st.metric("Features Analyzed", len(df.columns)-1)
    
    st.markdown("---")
    
    # Key Features Section
    st.header("🎯 Key Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - 📊 Comprehensive Data Analysis
        - 🔮 Real-time Prediction
        - 📈 Interactive Visualizations
        """)
    with col2:
        st.markdown("""
        - 🎯 High Accuracy
        - 🔍 Detailed Insights
        - 📱 Mobile Responsive
        """)

# Analysis Page
elif page == "📊 Analysis":
    st.title("📊 Data Analysis Dashboard")
    
    # Dataset Overview in Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Dataset Overview", "📊 Statistical Analysis", "🔍 Advanced Insights"])
    
    with tab1:
        st.subheader("📜 Dataset Overview")
        
        # Three columns for key metrics
        met1, met2, met3 = st.columns(3)
        with met1:
            st.metric("Total Records", f"{len(df):,}")
        with met2:
            st.metric("Features", f"{len(X.columns):,}")
        with met3:
            st.metric("Missing Values", "0")
        
        st.markdown("---")
        
        # Two columns for data display
        col1, col2 = st.columns([3,2])
        
        with col1:
            st.markdown("### Sample Data")
            st.dataframe(X.head(), use_container_width=True)
        
        with col2:
            st.markdown("### Statistical Summary")
            stats = X.describe().round(2)
            st.dataframe(stats, use_container_width=True)
            
            st.markdown("### Feature Types")
            feature_types = pd.DataFrame({
                'Feature': X.columns,
                'Type': X.dtypes,
                'Non-Null Count': X.count()
            })
            st.dataframe(feature_types, use_container_width=True)

    with tab2:
        st.subheader("📊 Statistical Analysis")
        
        # Two columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🔗 Feature Correlations")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax, fmt='.2f', annot_kws={'size': 8})
            plt.title("Correlation Matrix")
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📦 Summary Stats")
            st.dataframe(X.describe().T, use_container_width=True)

        st.markdown("---")

        # Box Plots
        st.markdown("### 📦 Data Distribution Analysis")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Raw Data Distribution")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, palette="Set2", ax=ax1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig1)
        
        with col4:
            st.markdown("#### Cleaned Data Distribution")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df1, palette="Set2", ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig2)

    with tab3:
        st.subheader("🔍 Advanced Insights")
        
        # Organizing Graphs into Sections
        st.markdown("### 🎯 Key Insights and Visualizations")

        # First Row: Pie Chart and Histogram-Density Combined
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.markdown("#### Diabetes Distribution (Pie Chart)")
            fig, ax = plt.subplots(figsize=(5, 4))
            percentage = y.value_counts() / len(y) * 100
            ax.pie(percentage, labels=["Non-Diabetic", "Diabetic"], autopct="%1.1f%%", colors=["green", "red"])
            ax.set_title("Diabetes Distribution")
            st.pyplot(fig)
        
        with row1_col2:
            st.markdown("#### Feature Distribution: Histogram and Density")
            hist_tab, density_tab = st.tabs(["📊 Histogram", "📈 Density Plot"])

            with hist_tab:
                fig, axes = plt.subplots(4, 2, figsize=(10, 8))
                axes = axes.flatten()
                for i, col in enumerate(X.columns):
                    X[col].hist(ax=axes[i], bins=20, color="skyblue", edgecolor="black")
                    axes[i].set_title(f"Histogram of {col}")
                plt.tight_layout()
                st.pyplot(fig)

            with density_tab:
                fig, axes = plt.subplots(4, 2, figsize=(10, 8))
                axes = axes.flatten()
                for i, col in enumerate(X.columns):
                    sns.kdeplot(X[col], ax=axes[i], color="blue")
                    axes[i].set_title(f"Density of {col}")
                plt.tight_layout()
                st.pyplot(fig)

        st.markdown("---")

        # Second Row: BMI Distribution and Age vs Pregnancies
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.markdown("#### BMI Distribution Analysis")
            fig, ax = plt.subplots(figsize=(5, 4))
            bmi_bins = [0, 18.5, 24.9, 29.9, 40, 100]
            bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese', 'Morbidly Obese']
            df['bmi_group'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels, right=False)
            sns.countplot(x='bmi_group', hue='Outcome', data=df, palette="Set2", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with row2_col2:
            st.markdown("#### Age vs Pregnancies (Scatter Plot)")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.scatterplot(data=df, x='Pregnancies', y='Age', hue='Outcome', palette='Set2', ax=ax)
            plt.title("Age vs Pregnancies Distribution")
            st.pyplot(fig)

        st.markdown("---")

        # Key Observations Section
        st.markdown("### 🔍 Key Observations")
        observations = [
            "People suffering from Diabetes are mostly in the Obese BMI category.",
            "Age group 21–40 shows the highest diabetes prevalence.",
            "Diabetic probability tends to decrease with increased pregnancies.",
            "Age correlates with number of pregnancies, but not necessarily with diabetes probability."
        ]
        for obs in observations:
            st.markdown(f"- {obs}")


# Prediction Page
elif page == "🔮 Prediction":
    st.title("🔮 Diabetes Risk Assessment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Patient Information")
        with st.form("prediction_form"):
            input_col1, input_col2 = st.columns(2)
            
            user_input = {}
            for i, col in enumerate(X.columns):
                with input_col1 if i % 2 == 0 else input_col2:
                    if col == "Pregnancies":  # Check if the column is "Pregnancies"
                        user_input[col] = st.slider(
                            f"{col}",
                            int(X[col].min()),  # Minimum value as integer
                            int(X[col].max()),  # Maximum value as integer
                            int(X[col].mean()),  # Default value as integer
                            step=1  # Step value to ensure only integers
                        )
                    else:
                        user_input[col] = st.slider(
                            f"{col}",
                            float(X[col].min()),
                            float(X[col].max()),
                            float(X[col].mean())
                        )
            
            submitted = st.form_submit_button("Predict")
    
    with col2:
        st.markdown("### Prediction Results")
        if submitted:
            input_df = pd.DataFrame([user_input])
            scaled_input = Standard_scaler.transform(input_df)
            prediction = logistic_model.predict(scaled_input)[0]
            probability = logistic_model.predict_proba(scaled_input)[0]
            
            if prediction == 1:
                st.error("#### Prediction: Diabetic")
                st.progress(probability[1])
                st.write(f"Confidence: {probability[1]:.2%}")
            else:
                st.success("#### Prediction: Non-Diabetic")
                st.progress(probability[0])
                st.write(f"Confidence: {probability[0]:.2%}")
            
            st.markdown("### Input Summary")
            st.dataframe(input_df)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Diabetes Prediction System</p>
    </div>
    """,
    unsafe_allow_html=True
)