import os
import numpy as np

from os.path import join

class Experiments():
    def __init__(self) -> None:
        self.TEST_SAMPLES = os.getenv('TEST_SAMPLES')
        self.TEST_DATASET_PATH = os.getenv('TEST_DATASET_PATH')
        self.TEST_DATASET_PATH_DESTINATION = os.getenv('TEST_DATASET_PATH_DESTINATION')

        self.experiment_1 = [[0, 330], [0, 330], [0, 330], [0, 10]] 
        self.experiment_2 = [[330, 646], [330, 646], [330, 647], [10, 60]]
        self.experiemnt_3 = [[646, 946], [646, 946], [647, 947], [60, 160]]
        self.experiment_4 = [[946, 1213], [946, 1213], [947, 1213], [160, 360]]
        self.experiment_5 = [[1213, 1380], [1213, 1379], [1213, 1379], [360, 860]]

        self.experiments = [self.experiment_1, self.experiment_2, self.experiemnt_3, self.experiment_4, self.experiment_5]

        self.dirs = ['experiment_1', 'experiment_2', 'experiment_3', 'experiment_4', 'experiment_5']

    def separate_samples_classes(self):
        for folder_name, experimet_images in zip(self.dirs, self.experiments):
            # create folder inside project
            os.makedirs(join(self.TEST_DATASET_PATH_DESTINATION, folder_name))

            # get all files in datasets
            files_in_dir = os.listdir(join(self.TEST_DATASET_PATH))

            for image_range in experimet_images:
                os.makedirs(join(self.TEST_DATASET_PATH_DESTINATION, folder_name, 'covid'))
                

if __name__ == '__main__':
    experiments = Experiments()
    experiments