import PIL.Image
import os
import glob
import numpy as np

from torch.utils.data import Dataset


class H36(Dataset):
    def __init__(self, root, train=True, range_fr=30, transform=None):
        super(H36, self).__init__()
        if train:
            self.root = os.path.join(root, 'train')
        else:
            self.root = os.path.join(root, 'test')

        self.train = train
        self.range_fr = range_fr
        self.transform = transform

        self.files = self._get_file_list()

    def _get_file_list(self):
        root = os.path.join(self.root, '**', '*.jpg')
        files = glob.glob(root, recursive=True)
        files = [f for f in files if f.endswith('jpg')]
        files = [f.split('/')[-3:] for f in files]

        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        num_fr = len(glob.glob(os.path.join(self.root, file[0], file[1], '*.jpg'), recursive=True))

        source = int(file[2].rstrip('.jpg'))
        target = np.random.randint(low=3, high=self.range_fr + 1)

        if source + target < num_fr:
            target = source + target
        else:
            target = source - target

        source = os.path.join(self.root, file[0], file[1], file[2])
        target = os.path.join(self.root, file[0], file[1], f'{target}.jpg')

        source = PIL.Image.open(source)
        target = PIL.Image.open(target)

        source = self.transform(source)
        target = self.transform(target)

        return source, target
