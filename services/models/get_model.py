import mlflow
import mlflow.sklearn
import joblib

MLFLOW_TRACKING_URI = r"sqlite:///D:\git repos\lab1\mlflow\mlruns.db"  # путь от models/ где лежат модели
RUN_ID = "8411f175e191450f9f63ddc64e975fea"
MODEL_PATH = "model.pkl"

def download_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{RUN_ID}/model"
    model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to current dir {MODEL_PATH}")

if __name__ == "__main__":
    download_model()