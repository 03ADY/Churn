# 🚀 Advanced AI Customer Churn Prediction

This project combines the best features from two previous churn prediction applications to deliver a comprehensive, interactive, and powerful tool for analyzing and predicting customer churn. It leverages both Random Forest and Neural Network models, provides rich data exploration, and offers live individual and batch prediction capabilities through a user-friendly Streamlit interface.

## ✨ Features

* **Interactive User Interface:** Built with Streamlit for a seamless and intuitive user experience.
* **Flexible Data Input:** Upload your own CSV dataset or use the automatically generated synthetic data for demonstration.
* **Dual Model Approach:** Trains and utilizes both a **Random Forest Classifier** and a **Neural Network** for robust predictions.
* **Comprehensive Model Evaluation:**
    * Interactive radar charts for comparing key performance metrics (Accuracy, Precision, Recall, F1 Score, AUC) for both models.
    * Detailed confusion matrices and ROC curves for visual performance analysis.
* **In-depth Feature Importance:** Visualizes both built-in (for Random Forest) and permutation importance to understand which factors drive churn predictions.
* **Live Individual Prediction:** Input specific customer details via an interactive form and get instant churn probabilities and risk levels from both models, along with a feature contribution breakdown.
* **Batch Prediction:** Upload a CSV file containing multiple customer profiles to get bulk predictions from both models, including risk levels, and download the results.
* **Interactive Data Explorer:** Visualize data distributions (age, balance, geography) and feature correlations with dynamic Plotly charts.
* **Enhanced UI/UX:** Custom CSS provides a modern and engaging look and feel.

## 🛠️ Technologies Used

* **Python 3.x**
* **Streamlit:** For building the interactive web application.
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Scikit-learn:** For data preprocessing (StandardScaler, OneHotEncoder, ColumnTransformer), model training (RandomForestClassifier, GridSearchCV), and evaluation metrics.
* **TensorFlow/Keras:** For building and training the Neural Network model.
* **Plotly:** For creating interactive and visually appealing data visualizations.
* **XGBoost:** (Included primarily for permutation importance calculation, though Random Forest is the tree-based model trained directly in the app).

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
    (Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub details.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

    Your browser should automatically open to the Streamlit application (usually at `http://localhost:8501`).

## ☁️ Deployment on Streamlit Community Cloud

1.  Push all the files (`app.py`, `requirements.txt`, `README.md`, `.gitignore`) to a **public GitHub repository**.
2.  Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in with your GitHub account.
3.  Click on "New app" -> "From existing repo".
4.  Select your repository, the branch, and set the main file path to `app.py`.
5.  Click "Deploy!"

Streamlit will automatically build and deploy your application, providing you with a public URL to share.

## 📊 Data Format for Upload

If you choose to upload your own data for training or batch prediction, ensure your CSV file contains the following columns (case-sensitive):

* `CreditScore` (int)
* `Geography` (string: 'France', 'Spain', 'Germany')
* `Gender` (string: 'Male', 'Female')
* `Age` (int)
* `Tenure` (int)
* `Balance` (float)
* `EstimatedSalary` (float)
* `Exited` (int: 0 for 'Stay', 1 for 'Churn' - **only required for training data**)

Columns like `RowNumber`, `CustomerId`, and `Surname` will be automatically ignored if present. A sample format is provided within the app's "Batch Prediction" tab for download.

---
