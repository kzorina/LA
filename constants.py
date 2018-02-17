DATA_DIR = "yalefaces"
TRAIN_DIR = "train_{}".format(DATA_DIR)
TEST_DIR = "test_{}".format(DATA_DIR)
SCALE_PARAMETER = 0.25
STANDARD_SIZE = (243, 320)
N_COMPONENTS = 20
SVD_SOLVER = "auto"
WHITEN = True
IMAGE_FEATURES = ["centerlight", "glasses", "happy", "leftlight", "noglasses", "normal",
                  "rightlight", "sad", "sleepy", "surprised", "wink"]
GENERIC_IMAGENAME = "subject{number}.{feature}"
N_OF_EVCS_TO_DROP = 1
N_OF_FOLDS = 5
NUMBER_OF_PEOPLE = 15
