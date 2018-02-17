import os
from shutil import copyfile


def remove_content_from_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return

    for the_file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)


def copy_image(index, feature, generic_name, src, dst):
    person_name = str(index + 1).rjust(2, "0")
    full_name = generic_name.format(number=person_name, feature=feature)
    path_from = os.path.join(src, full_name)
    path_to = os.path.join(dst, full_name)
    copyfile(path_from, path_to)


def split_images(fold_index, generic_name, features, number_of_people, number_of_folds, src,
                 train_dst, test_dst):
    begin_feature = int(len(features) * fold_index / number_of_folds)
    end_feature = int(len(features) * (fold_index + 1) / number_of_folds)
    for index in range(number_of_people):
        for feature in (features[0:begin_feature]):
            copy_image(index, feature, generic_name, src, train_dst)

        for feature in (features[begin_feature:end_feature]):
            copy_image(index, feature, generic_name, src, test_dst)

        for feature in (features[end_feature:len(features)]):
            copy_image(index, feature, generic_name, src, train_dst)
