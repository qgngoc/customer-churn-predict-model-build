import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_path):
    # Load the dataset
    data = pd.read_csv(input_path)

    # Handle missing values
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

    # Drop 'customerID' as it is not useful for prediction
    if 'customerID' in data.columns:
        data.drop('customerID', axis=1, inplace=True)

    # Convert 'SeniorCitizen' to a categorical feature
    if 'SeniorCitizen' in data.columns:
        data['SeniorCitizen'] = data['SeniorCitizen'].astype('category')

    # Replace 'No internet service' and 'No phone service' with 'No' for consistency
    columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for column in columns_to_replace:
        if column in data.columns:
            data[column] = data[column].replace({'No internet service': 'No'})
    if 'MultipleLines' in data.columns:
        data['MultipleLines'] = data['MultipleLines'].replace({'No phone service': 'No'})

    # Encode categorical variables
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                           'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

    # Filter categorical columns to include only those present in the dataset
    categorical_columns = [col for col in categorical_columns if col in data.columns]

    # Apply one-hot encoding
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Convert all True/False columns to 1/0
    bool_columns = data.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        data[col] = data[col].astype(int)

    # Feature scaling
    scaler = StandardScaler()
    data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(data[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # Define features and target variable
    X = data.drop('Churn_Yes', axis=1)
    y = data['Churn_Yes']

    # Split into train and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Further split the test set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Combine features and labels for saving
    train = pd.concat([y_train, X_train], axis=1)
    val = pd.concat([y_val, X_val], axis=1)
    test = pd.concat([y_test, X_test], axis=1)

    return train, val, test

if __name__ == "__main__":
    input_path = "/opt/ml/processing/input/churn.csv"

    train_data, val_data, test_data = preprocess_data(input_path=input_path)
    try:
        os.makedirs("/opt/ml/processing/train", exist_ok=True)
        os.makedirs("/opt/ml/processing/validation", exist_ok=True)
        os.makedirs("/opt/ml/processing/test", exist_ok=True)
    except Exception as e:
        pass

    train_data.to_csv("/opt/ml/processing/train/train.csv", header=False, index=False)
    val_data.to_csv("/opt/ml/processing/validation/validation.csv", header=False, index=False)
    test_data.to_csv("/opt/ml/processing/test/test.csv", header=False, index=False)

    # # Test
    # train_data, val_data, test_data = preprocess_data(input_path="../data/raw/churn.csv")
    # os.makedirs("../data/processed/train", exist_ok=True)
    # os.makedirs("../data/processed/validation", exist_ok=True)
    # os.makedirs("../data/processed/test", exist_ok=True)
    # train_data.to_csv("../data/processed/train/train.csv", header=False, index=False)
    # val_data.to_csv("../data/processed/validation/validation.csv", header=False, index=False)
    # test_data.to_csv("../data/processed/test/test.csv", header=False, index=False)

# Example usage
# preprocess_data("../data/raw/churn.csv", "../data/processed/")