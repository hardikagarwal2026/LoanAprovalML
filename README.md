# Loan Approval Prediction

## Overview
This project predicts whether a loan application will be approved based on applicant details using machine learning techniques. It leverages the **AdaBoost algorithm** and deploys the trained model using the **ONNX framework** for efficient inference.

## Features
- **Loan Eligibility Prediction**: Predicts loan approval status based on applicant details.
- **Machine Learning Model**: Utilizes the **AdaBoost classifier** for better accuracy.
- **ONNX Deployment**: Enables lightweight and scalable inference.
- **Data Preprocessing**: Handles missing values and encodes categorical features.
- **Exploratory Data Analysis (EDA)**: Provides insights into loan approval trends.
- **Performance Metrics**: Evaluates model accuracy, precision, recall, and F1-score.

## Dataset
- **File Name**: `Loan-Approval-Prediction.csv`
- **Description**: Contains loan application data with features like income, loan amount, credit history, property area, and more.
- **Preprocessing Steps**:
  - Handling missing values in categorical and numerical columns.
  - Encoding categorical variables using label encoding.
  - Normalizing numerical features to improve model performance.

## Installation & Setup
1. **Clone the repository**:
   ```sh
   git clone https://github.com/hardikagarwal2026/LoanAprovalML.git
   cd LoanAprovalML-main
   ```
2. **Create a virtual environment (optional but recommended)**:
   ```sh
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows
   ```
3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
4. **Launch Jupyter Notebook**:
   ```sh
   jupyter notebook
   ```
5. **Open and run**:
   - `Loan_price_prediction.ipynb` (to train and evaluate the model)
   - `get_prediction_using_onnx.ipynb` (to test the trained model)

## Model Training
- **Algorithm Used**: `AdaBoost Classifier`
- **Training Process**:
  - Splits the dataset into training and testing sets.
  - Trains the AdaBoost classifier on processed data.
  - Evaluates the model using performance metrics.
- **Model Export**: The trained model is converted to the ONNX format for optimized inference.
- **Model File**: `adaboost_loan_model.onnx`

## Usage
### Training the Model
1. Open `Loan_price_prediction.ipynb` in Jupyter Notebook.
2. Run the cells to preprocess data, train the model, and evaluate performance.
3. The final model will be saved as `adaboost_loan_model.onnx`.

### Making Predictions Using ONNX
1. Open `get_prediction_using_onnx.ipynb`.
2. Load the `adaboost_loan_model.onnx` file.
3. Input new loan applicant details and get predictions.

## Performance Metrics
- **Accuracy Score**: Measures overall correctness.
- **Precision & Recall**: Evaluates false positives and false negatives.
- **F1-Score**: Balances precision and recall.
- **Confusion Matrix**: Provides a breakdown of prediction results.

## Dependencies
- Python 3.8+
- Scikit-learn
- ONNX
- Pandas
- NumPy
- Matplotlib (for data visualization)
- Jupyter Notebook

## Future Enhancements
- Integrating deep learning models for better accuracy.
- Deploying as a web application using Flask or FastAPI.
- Implementing real-time loan approval predictions.

## Contributors
- Hardik Agarwal
- Open to contributors! Feel free to fork and contribute via pull requests.
