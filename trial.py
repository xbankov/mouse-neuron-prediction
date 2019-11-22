from pathlib import Path

import numpy as np
from PIL import Image
from pandas import DataFrame
from tqdm import tqdm


class Trial:
    def __init__(self, spatial, responses, labels, classes, filename):
        self.labels = self.get_labels(labels, classes)
        self.xs = spatial[:, 0]
        self.ys = spatial[:, 1]
        self.zs = spatial[:, 2]
        self.trial_name = self.get_trial_name(filename)
        self.images_names = self.get_image_names(responses)

        if not self.exists_images():
            self.images = self.get_blended_images(responses)

    @staticmethod
    def get_trial_name(filename):
        return filename.split('/')[-1].split('.')[-2]

    @staticmethod
    def exists_image(filename):
        return Path(filename).is_file()

    @staticmethod
    def get_labels(labels, classes):
        labels = DataFrame(labels).apply(lambda row: classes.iloc[int(row[0]) - 1, 0], axis=1)
        return labels.to_numpy()

    @staticmethod
    def blend_images(images):
        return (np.clip(sum(images), 0, 255)).astype('uint8')

    def get_labeling_df(self):
        return DataFrame({"name": self.images_names, "label": self.labels})

    def exists_images(self):
        for name in self.images_names:
            if not self.exists_image(name):
                return False
        return True

    def get_blended_images(self, responses):
        uniques = np.unique(self.zs)
        images = []
        # TODO Run on all samples from trial
        for sample in tqdm(range(responses.shape[0])):
            images.append((self.blend_images([self.make_img_array(unique, responses, sample) for unique in uniques])))
        return np.array(images)

    def make_img_array(self, unique, responses, sample):
        mask = np.where(self.zs == unique)
        xs_ = self.xs[mask]
        ys_ = self.ys[mask]
        c_ = responses[sample, :][mask]

        arr = np.zeros((max(self.xs) + 1, max(self.ys) + 1))
        for i in range(len(c_)):
            arr[xs_[i]][ys_[i]] = c_[i]
        return arr

    def get_image_names(self, responses, folder='data'):
        return [f"{folder}/{i}_{self.trial_name}.png" for i in range(responses.shape[0])]

    def save_blended_images(self):
        if not self.exists_images():
            for i, image in enumerate(self.images):
                img = Image.fromarray(image, 'L')
                img.save(self.images_names[i])
