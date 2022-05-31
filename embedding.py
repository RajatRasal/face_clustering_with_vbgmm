import argparse

import mlflow
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from functools import partial
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
from tqdm import tqdm

from dataset_by_person import CelebAByPerson, random_ids_celeba, most_common_ids_celeba


EMBEDDING_ARTIFACT_NAME = 'face_det_and_embed.npy'

def face_detection_and_embedding(dataset, embedder, batch_size=32, margin=20):
    # Note: Assuming that all the datasets we're using have 1 face per image
    mtcnn = MTCNN(margin=margin, select_largest=False).eval() 

    total = len(dataset)
    all_ids = []
    all_faces = []
    all_embeddings = []
    
    for i in tqdm(range(0, total, batch_size)):
        lb = i
        ub = min(i + batch_size, total)

        imgs_batch = []
        ids_batch = []
        for j in range(lb, ub):
            ids_batch.append(int(dataset[j][1]))
            imgs_batch.append(dataset[j][0])
            
        faces = mtcnn(imgs_batch)

        for _id, face in zip(ids_batch, faces):
            if face is not None:
                all_embeddings.append(embedder(face.unsqueeze(0)).detach().numpy())
                all_faces.append(face.detach().numpy())
                all_ids.append(_id)

    return {
        'ids': all_ids,
        'faces': all_faces,
        'embeddings': all_embeddings,
    }

def get_artifact_run(tags):
    query = [
        f"tag.`{key}` = '{value}'"
        for key, value in tags.items()
    ]
    query = " and ".join(query)

    all_experiments = [
        exp.experiment_id
        for exp in MlflowClient().list_experiments()
    ]
    runs = MlflowClient().search_runs(
        experiment_ids=all_experiments,
        filter_string=query,
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    return runs

def main(
    dataset: str,
    split: str,
    identities: int,
    selection_method: str,
    random_seed: int,
    embedder: str,
    batch_size: int,
    margin: int
):
    datasets = {
        'CelebA': (
            CelebAByPerson,
            {
                'random': partial(random_ids_celeba, seed=random_seed),
                'topk': most_common_ids_celeba,
            }
        )
    }

    ds_cls, selectors = datasets[dataset]
    ids = selectors[selection_method](root='.', top_k=identities)
    ds = ds_cls(ids=ids, root='.')

    embedders = {
        'facenet_vggface2': InceptionResnetV1(pretrained="vggface2").eval(),
        'facenet_webface': InceptionResnetV1(pretrained="casia-webface").eval(),
    }
    embedding_net = embedders[embedder]

    tags = {
        "dataset": dataset,
        "split": split,
        "identities": identities,
        "selection_method": selection_method,
        "random_seed": random_seed,
        "embedder": embedder,
    }

    if len(get_artifact_run(tags)) != 0:
        print('Artifact exists')
        return

    results = face_detection_and_embedding(
        ds,
        embedding_net,
        batch_size=batch_size,
        margin=margin
    )
    np.save(EMBEDDING_ARTIFACT_NAME, results)

    with mlflow.start_run(run_name="Dataset"):
        mlflow.set_tags(tags)
        mlflow.log_artifact(EMBEDDING_ARTIFACT_NAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Embeddings arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["CelebA"],
        nargs="?",
        default="CelebA",
        help="Name of dataset."
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "val"],
        nargs="?",
        default="train",
        help="Split for dataset."
    )
    parser.add_argument(
        "--identities",
        type=int,
        default=50,
        help="Number of identities to randomly select."
    )
    parser.add_argument(
        "--selection_method",
        type=str,
        choices=["random", "topk"],
        default="topk",
        help="Number of identities to randomly select."
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for identity selection."
    )
    parser.add_argument(
        "--embedder",
        type=str,
        choices=["facenet_vggface2", "facenet_webface"],
        default="facenet_vggface2",
        help="MLFlow model name for image embedding network."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for face detection"
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=20,
        help="Margin around each face"
    )

    args = parser.parse_args()
    main(
        args.dataset,
        args.split,
        args.identities,
        args.selection_method,
        args.random_seed,
        args.embedder,
        args.batch_size,
        args.margin,
    )
