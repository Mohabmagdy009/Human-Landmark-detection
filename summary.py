from model import build_landmark_model
from config import *

# Create the Model
model = build_landmark_model(input_shape=input_shape, output_size=num_marks*2)

print(model.summary())
