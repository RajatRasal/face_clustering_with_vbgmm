import concurrent
from tqdm import tqdm
from typing import Set
from torchvision.datasets import CelebA


class CelebAByPerson(CelebA):

    def __init__(self, ids: Set[int], **kwargs):
        kwargs['download'] = False
        super().__init__(**kwargs)

        self.target_type = ['attr', 'identity', 'bbox', 'landmarks']
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


if __name__ == "__main__":
    dataset = CelebAByPerson(people_ids=[1,2,3], root='.')
    print(len(dataset))
