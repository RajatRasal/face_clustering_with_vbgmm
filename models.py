import mlflow
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

import metrics
from embedding import get_artifact_run, EMBEDDING_ARTIFACT_NAME

params = {
    "n_components": 200,
    "random_state": 42,
    "verbose": 2,
    "weight_concentration_prior": 10,
}

tags = {
    "dataset": "CelebA",
    "split": "train",
    "identities": 50,
    "selection_method": "topk",
    "random_seed": 42,
    "embedder": "facenet_vggface2",
}

# TODO: Make experiment with these dataset tags and VBGMM

# TODO: Put all datasets under the same experiment
run = get_artifact_run(tags)[0]
artifact = mlflow.artifacts.download_artifacts(
    artifact_uri=run.info.artifact_uri,
)
data = np.load(f"{artifact}/{EMBEDDING_ARTIFACT_NAME}", allow_pickle=True)[()]

embeddings_arr = np.concatenate(data["embeddings"])

def train_model(params):
    vbgmm = BayesianGaussianMixture(**params)
    mlflow.sklearn.log_model(vbgmm, "model")
    clusters = vbgmm.fit_predict(embeddings_arr)
    return clusters

mlflow.sklearn.autolog()
with mlflow.start_run(run_name="vbgmm"):
    # TODO: GridSearch
    # Automatically logs hyperparameters
    clusters = train_model(params)

true_clusters = np.array(data["ids"])
_metrics = metrics.metrics(true_clusters, clusters)
print(_metrics)
jsd = metrics.jensen_shannon_distance(
    true_clusters, clusters, tags["identities"]
)
print(jsd)