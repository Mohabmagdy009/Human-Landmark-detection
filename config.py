# Checkpoint is used to track the training model so it could be restored
# later to resume training.
# checkpoint_dir = "checkpoints/upper"
checkpoint_dir = "checkpoints/full"

# Besides checkpoint, `saved_model` is another way to save the model for
# inference or optimization.
# export_dir = "exported/upper"
export_dir = "exported/full"

# The log directory for tensorboard.
# log_dir = "logs/upper"
log_dir = "logs/full"

# The input image's width, height and channels should be consist with your
# training data. Here they are set to be complied with the tutorial.
input_shape = (128, 128, 3)

# custom landmarks
# num_marks = 12
# LANDMARK_NAMES = ["L.Chest", "R.Chest", "Shoulder.A", "Shoulder.B", "Shoulder.C", "Shoulder.D", "Shoulder.E", "Arm.A", "Arm.B", "L.Waist", "R.Waist", "Arm.E"]

# upper-body landmarks
# num_marks = 6
# LANDMARK_NAMES = ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]

# lower-body landmarks
# num_marks = 4
# LANDMARK_NAMES = ["left waistline", "right waistline", "left hem", "right hem"]

# full-body landmarks
num_marks = 8
LANDMARK_NAMES = ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline", "right waistline", "left hem", "right hem"]
