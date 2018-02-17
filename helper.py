import os
from shutil import copyfile
from constants import *


def remove_content_from_dir(dir_path):
    for the_file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)

def copy_image(index, feature, dst):
    person_name = str(index + 1).rjust(2, "0")
    full_name = GENERIC_IMAGENAME.format(number=person_name, feature=feature)
    path_from = os.path.join(DATA_DIR, full_name)
    path_to = os.path.join(dst, full_name)
    copyfile(path_from, path_to)

def split_images(fold_index):
    begin_feature = int(len(IMAGE_FEATURES) * fold_index / N_OF_FOLDS)
    end_feature = int(len(IMAGE_FEATURES) * (fold_index + 1) / N_OF_FOLDS)
    for index in range(NUMBER_OF_PEOPLE):
        for feature in (IMAGE_FEATURES[0:begin_feature]):
            copy_image(index, feature, TRAIN_DIR)

        for feature in (IMAGE_FEATURES[begin_feature:end_feature]):
            copy_image(index, feature, TEST_DIR)

        for feature in (IMAGE_FEATURES[end_feature: len(IMAGE_FEATURES)]):
            copy_image(index, feature, TRAIN_DIR)
