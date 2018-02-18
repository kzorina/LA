import time

from constants import DATA_DIR, GENERIC_IMAGENAME, IMAGE_FEATURES, NUMBER_OF_PEOPLE, N_COMPONENTS, \
    N_OF_EVCS_TO_DROP, N_OF_FOLDS, SCALE_PARAMETER, STANDARD_SIZE, TEST_DIR, TRAIN_DIR
from eigenface import Eigenface
from helper import remove_content_from_dir, split_images
import matplotlib.pyplot as plt


eigenface = Eigenface()

results = []
t_global = time.time()
for fold_index in range(N_OF_FOLDS):
    t_per_fold = time.time()
    print("Fold number {}".format(fold_index + 1))

    remove_content_from_dir(TRAIN_DIR)
    remove_content_from_dir(TEST_DIR)
    split_images(fold_index, GENERIC_IMAGENAME, IMAGE_FEATURES, NUMBER_OF_PEOPLE, N_OF_FOLDS,
                 DATA_DIR, TRAIN_DIR, TEST_DIR)

    face_matrix = eigenface.read_data(TRAIN_DIR, SCALE_PARAMETER)
    mean_face = eigenface.get_mean(face_matrix)
    eigenface.normalize(face_matrix, mean_face)
    covariance_matrix = eigenface.get_covariance_matrix(face_matrix)
    svd = eigenface.get_svd(covariance_matrix, N_COMPONENTS)
    # for index, evc in enumerate(svd.components_):
    #     if index < 4:
    #         plt.figure()
    #         plt.imshow(evc.reshape(int((STANDARD_SIZE[0] + 1) / 4), int(STANDARD_SIZE[1] / 4)),
    #                    cmap=plt.cm.gray)
    #         plt.show()
    svd.components_ = svd.components_[N_OF_EVCS_TO_DROP:]



    decomposed_images = eigenface.train_model(
        TRAIN_DIR, mean_face, SCALE_PARAMETER, svd, STANDARD_SIZE)
    matches = eigenface.recognize_test_images(
        TEST_DIR, mean_face, SCALE_PARAMETER, svd, decomposed_images, STANDARD_SIZE)

    accuracy = matches.count(True) / len(matches)
    results.append(accuracy)
    print("Accuracy: {}".format(accuracy))
    print("Per-fold time: {}".format(time.time() - t_per_fold))

print("Overall accuracy: {}".format(sum(results) / len(results)))
print("Global time: {}".format(time.time() - t_global))
