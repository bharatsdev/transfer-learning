import os

import numpy as np
import cv2


class SimpleDatasetLoader:
    def __init__(self, preprocessor=None):
        if preprocessor:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = []

    def load(self, imgPaths, verbose=-1):
        data = []
        labels = []
        for (idx, imagePath) in enumerate(imgPaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            for p in self.preprocessor:
                image = p.preprocessor(image)
            data.append(image)
            labels.append(label)
            # show an update every `verbose` images
        if verbose > 0 and idx > 0 and (idx + 1) % verbose == 0:
            print("[INFO] : Processed {}/{}".format(idx + 1, len(imgPaths)))
        return np.array(data), np.array(labels)
