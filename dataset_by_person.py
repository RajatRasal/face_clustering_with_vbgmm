import random
from collections import Counter

from torchvision.datasets import CelebA
from typing import Set


class CelebAByPerson(CelebA):
    def __init__(self, ids: Set[int], **kwargs):
        # TODO: Download all the CelebA data from Google Drive
        kwargs['download'] = False
        super().__init__(**kwargs)

        self.target_type = ['identity']
        self.ids = ids

        self.idxs = [
            i
            for i in range(len(self.identity))
            if int(self.identity[i][0]) in self.ids
        ]

    def __getitem__(self, idx: int):
        return super().__getitem__(self.idxs[idx])

    def __len__(self) -> int:
        return len(self.idxs)

def most_common_ids_celeba(root='.', top_k=50):
    dataset = CelebA(root=root)
    identities = dataset.identity.flatten().tolist()
    chosen_ids = [_id for _id, _ in Counter(identities).most_common(top_k)]
    return chosen_ids

def random_ids_celeba(root='.', top_k=50, seed=42):
    dataset = CelebA(root=root)
    identities = dataset.identity.flatten().tolist()
    chosen_ids = random.sample(identities, top_k, seed=seed)
    return chosen_ids


if __name__ == "__main__":
    dataset = CelebAByPerson(people_ids=[1,2,3], root='.')
    print(len(dataset))
