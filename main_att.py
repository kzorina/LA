from constants_att import N_COMPONENTS, N_OF_EVCS_TO_DROP, N_OF_FOLDS, SCALE_PARAMETER, \
    STANDARD_SIZE, TEST_DIR, TRAIN_DIR, GENERIC_IMAGENAME, IMAGE_FEATURES, NUMBER_OF_PEOPLE, \
    DATA_DIR
from eigenface import Eigenface
from helper import remove_content_from_dir, split_images

eigenface = Eigenface()

results = []
for fold_index in range(N_OF_FOLDS):
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
    svd.components_ = svd.components_[N_OF_EVCS_TO_DROP:]

    decomposed_images = eigenface.train_model(
        TRAIN_DIR, mean_face, SCALE_PARAMETER, svd, STANDARD_SIZE)
    matches = eigenface.recognize_test_images(
        TEST_DIR, mean_face, SCALE_PARAMETER, svd, decomposed_images, STANDARD_SIZE)

    accuracy = matches.count(True) / len(matches)
    print("Accuracy: {}".format(accuracy))
    results.append(accuracy)

print("Overall accuracy: {}".format(sum(results) / len(results)))
