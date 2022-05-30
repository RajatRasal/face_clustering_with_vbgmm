import mlflow
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from dataset_by_person import init_dataset_most_common


params = {
    "n_components": 200,
    "random_state": 42,
    "verbose": 2,
    "weight_concentration_prior": 10,
}

# TODO: Use MLflow to mark dataset as artifact 
embeddings_arr = np.load('embeddings.npy')

def train_model(params):
    dataset = init_dataset_most_common()
    vbgmm = BayesianGaussianMixture(**params)
    mlflow.sklearn.log_model(vbgmm, "model")
    clusters = vbgmm.fit_predict(embeddings_arr)

mlflow.sklearn.autolog()
with mlflow.start_run(run_name="vbgmm"):
    train_model(params)
