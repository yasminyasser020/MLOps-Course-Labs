"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow
import joblib
def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    col_transf_path = "column_transformer.pkl"
    joblib.dump(col_transf, col_transf_path)
    mlflow.log_artifact(col_transf_path)
    



    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = mlflow.models.infer_signature(X_train, log_reg.predict(X_train))

    # Log model

   
    mlflow.sklearn.log_model(sk_model=log_reg, artifact_path="logistic_regression_model", signature=signature)
    ### Log the data
    
    temp_df = X_train.copy()
    temp_df['target'] = y_train.values
    train_dataset = mlflow.data.from_pandas(temp_df, targets='target', name="training_set")
    mlflow.log_input(train_dataset, context="training")


    return log_reg

def train_rf(X_train, y_train, n_estimators=100, max_depth=None):
   
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
   
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("model_type", "RandomForest")
    
    signature = mlflow.models.infer_signature(X_train, rf.predict(X_train))
    mlflow.sklearn.log_model(rf, "random_forest_model", signature=signature)

    return rf
from sklearn.svm import SVC

def train_svm(X_train, y_train, C=1.0, kernel='rbf'):
    
    svm_model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
  
    mlflow.log_param("C", C)
    mlflow.log_param("kernel", kernel)
    mlflow.log_param("model_type", "SVM")
    
    # 2. Infer Signature
    signature = mlflow.models.infer_signature(X_train, svm_model.predict(X_train))
    
    # 3. Log Model
    mlflow.sklearn.log_model(svm_model, "svm_model", signature=signature)
    
    return svm_model


def main():
    ### Set the tracking URI for MLflow

    mlflow.set_tracking_uri("http://localhost:5000")

    ### Set the experiment name
    
    exp_id = mlflow.set_experiment("First Experiment")
    runs = [
        {"name": "Logistic_Regression", "func": train, "params": {}},
        {"name": "Random_Forest", "func": train_rf, "params": {"n_estimators": 200, "max_depth": 15}},
        {"name": "SVM", "func": train_svm, "params": {"C": 1.5, "kernel": "rbf"}} ]


    ### Start a new run and leave all the main function code as part of the experiment
    for run in runs:
     with mlflow.start_run(run_name=run["name"]):

        df = pd.read_csv(r"D:\ITI\virtual_environment\MLOps-Course-Labs\dataset\Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)
       

        ### Log the max_iter parameter
        

        model = run["func"](X_train, y_train,**run["params"])
        mlflow.log_param("max_iter", 1000)
        

        
        y_pred = model.predict(X_test)

        ### Log metrics after calculating them
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1", f1_score(y_test, y_pred))



        ### Log tag
        mlflow.set_tag("model_type", f"{run["name"]}")

        
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        
        # Log the image as an artifact in MLflow
        plt.title(f"{run["name"]}_Confusion Matrix")
        plot_path = f"{run["name"]}confusion_matrix.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        
        plt.show()

if __name__ == "__main__":
    main()
