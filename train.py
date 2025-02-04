import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

# Set MLflow environment variables
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/abhay.optimus.2727/Machinelearningpipelines.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "abhay.optimus.2727"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c471037a9fd47db7a47c857abc0765b1abb55ad5"

# Function to perform hyperparameter tuning
def hyperparameter_tuning(x_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_searchCV = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_searchCV.fit(x_train, y_train)
    return grid_searchCV

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_path, model_path, random_state, n_estimators, max_depth):
    # Load dataset
    data = pd.read_csv(data_path)
    x = data.drop(columns=["Outcome"])
    y = data["Outcome"]  # Fixed issue

    mlflow.set_tracking_uri("https://dagshub.com/abhay.optimus.2727/Machinelearningpipelines.mlflow")

    # Start MLflow experiment
    with mlflow.start_run():
        X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=random_state)
        signature = infer_signature(X_train, y_train)

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        grid_searchCV = hyperparameter_tuning(X_train, y_train, param_grid)  # Fixed incorrect variable

        best_model = grid_searchCV.best_estimator_  # Fixed typo

        # Model evaluation
        y_pred = best_model.predict(x_test)
        acc_score = accuracy_score(y_test, y_pred)  # Renamed variable to avoid conflict
        print(f"Accuracy: {acc_score}")

        # Log metrics and parameters in MLflow
        mlflow.log_metric("accuracy", acc_score)
        mlflow.log_param("best_n_estimators", grid_searchCV.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_searchCV.best_params_['max_depth'])
        mlflow.log_param("best_min_samples_split", grid_searchCV.best_params_['min_samples_split'])
        mlflow.log_param("best_min_samples_leaf", grid_searchCV.best_params_['min_samples_leaf'])

        # Log confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store == "file":  # Fixed comparison
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best_model")
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # Save the model locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(best_model, open(model_path, 'wb'))

        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train(params["data"], params["model"], params["random_state"], params["n_estimators"], params["max_depth"])