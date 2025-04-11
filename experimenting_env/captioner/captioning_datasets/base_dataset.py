import os
import cv2
import numpy as np

from torch.utils.data import Dataset as BaseDataset, DataLoader
from tqdm import tqdm


class CaptioningBaseDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids_img = os.listdir(images_dir)

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids_img]
        self.images_fps.sort()

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        filename = os.path.basename(self.images_fps[i])

        # Read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample_final = {'rgb': image.copy(), 'image': None, 'filename': filename}

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        sample_final['rgb'] = image.copy()

        # Apply preprocessing
        if self.preprocessing:
            image = self.preprocessing(image)
        sample_final['image'] = image
        return sample_final

    def __len__(self):
        return len(self.ids_img)


if __name__ == '__main__':
    dir_data = "..."
    RGB_DIR = os.path.join(dir_data, "rgb")
    dataset = CaptioningBaseDataset(RGB_DIR, augmentation=None)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Visualise some samples
    for i, sample_batch in enumerate(tqdm(dataset_loader)):
        # Load filename
        filename = sample_batch['filename'][0]
        # Load data
        image = sample_batch['rgb'].cpu().detach().numpy()[0].astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow("RGB", image)
        cv2.waitKey(0)