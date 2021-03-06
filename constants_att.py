DATA_DIR = "att_faces"
TRAIN_DIR = "train_{}".format(DATA_DIR)
TEST_DIR = "test_{}".format(DATA_DIR)
SCALE_PARAMETER = 1
STANDARD_SIZE = (112, 92)
N_COMPONENTS = 20
SVD_SOLVER = "auto"
WHITEN = True
IMAGE_FEATURES = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
GENERIC_IMAGENAME = "{number}_{feature}.pgm"
N_OF_EVCS_TO_DROP = 1
N_OF_FOLDS = 5
NUMBER_OF_PEOPLE = 40