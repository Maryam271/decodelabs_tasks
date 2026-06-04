** 📊 AI-Powered Data Analysis & Machine Learning Dashboard**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Machine Learning](https://img.shields.io/badge/ML-ScikitLearn-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**📌 Project Overview**

This project is a **full-stack Data Science and Machine Learning web application** built using Streamlit. It enables users to upload datasets and perform an end-to-end data science workflow including data preprocessing, exploratory data analysis (EDA), visualization, and predictive modeling in a single interactive dashboard.

The system is designed to simulate real-world data analytics pipelines used in industry environments.

---

**🎯 Objectives**

- Automate end-to-end data analysis workflow
- Provide interactive data exploration without coding
- Implement real-time data cleaning and preprocessing
- Build predictive models using machine learning
- Enhance decision-making through visualization and insights

**🚀 Key Features**

**📂 Data Handling**
- CSV file upload and dynamic dataset loading
- Raw vs cleaned dataset management using session state

**🧹 Data Cleaning Pipeline**
- Missing value imputation (mean/mode strategy)
- Duplicate record removal
- Categorical feature encoding (Label Encoding)
- Automated preprocessing workflow

**📊 Exploratory Data Analysis (EDA)**
- Statistical summary (mean, median, standard deviation, min, max)
- Correlation matrix analysis
- Outlier detection using boxplots
- Automated insights generation

**📈 Data Visualization**
- Histogram analysis
- Bar charts for categorical distribution
- Scatter plots for feature relationships
- Line charts for trends
- Box plots for distribution analysis
- Correlation heatmaps

**🤖 Machine Learning Model**
- Linear Regression implementation
- Train-test split (configurable)
- Model evaluation metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Actual vs Predicted visualization
- Regression fit analysis
- Feature importance via coefficients

**🛠️ Tech Stack**

- Python
- Streamlit (Frontend + Web App)
- Pandas, NumPy (Data Processing)
- Matplotlib, Seaborn (Visualization)
- Scikit-learn (Machine Learning)
- 
**🧠 Skills Demonstrated**

- Data Preprocessing & Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Visualization & Storytelling
- Machine Learning (Regression Models)
- Model Evaluation & Performance Metrics
- End-to-End ML Pipeline Development
- Web App Development using Streamlit
- Data-driven Decision Making

**🏗️ System Architecture**
User Upload (CSV)
↓
Data Ingestion Layer
↓
Preprocessing & Cleaning Module
↓
EDA & Visualization Engine
↓
Machine Learning Model (Regression)
↓
Evaluation & Insights Dashboard

 📊 Project Modules

- `app.py` → Main Streamlit application
- Data Upload Module
- Data Cleaning Module
- Dataset Analysis Module
- Visualization Module
- Prediction Module

📈 Model Performance Metrics

- MAE → Measures average prediction error
- MSE → Penalizes larger errors
- RMSE → Interpretable error metric
- R² Score → Model accuracy & fit quality

**⚙️ How to Run**

bash
git clone <repository-url>
cd <project-folder>
pip install -r requirements.txt
 streamlit run app.py
