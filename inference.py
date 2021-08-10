from tensorflow import keras
from config import *
import imageio
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

example = "data/auged/val_a1_1heavy_0010.jpg,4.82E+01,4.16E+01,1.25E+02,4.23E+01,4.31E+01,1.76E+01,6.00E+01,1.18E+01,8.41E+01,1.79E+00,1.15E+02,1.38E+01,1.30E+02,1.83E+01,2.26E+01,6.12E+01,4.46E+01,7.22E+01,5.29E+01,1.07E+02,1.23E+02,1.06E+02,1.42E+02,1.58E+02"


if __name__ == '__main__':
    # Load the Model
    model = keras.models.load_model(export_dir)

    while True:
        image_path = input('input image path:')
        img = imageio.imread(image_path)
        orig_size = img.shape
        img_resized = resize(img, (input_shape[0], input_shape[1]))

        ratio_x = input_shape[1] / orig_size[1]
        ratio_y = input_shape[0] / orig_size[0]

        prediction = model(np.expand_dims(img_resized, 0)).numpy().squeeze()
        prediction = [float(t) / ratio_x if idx % 2 == 0 else float(t) / ratio_y for idx, t in enumerate(prediction)]

        plt.imshow(img, zorder=1)

        deltaX = img.shape[0] * 0.005
        deltaY = img.shape[1] * 0.04

        for idx in range(len(LANDMARK_NAMES)):
            x = prediction[idx * 2]
            y = prediction[idx * 2 + 1]
            plt.text(x - deltaX, y - deltaY, LANDMARK_NAMES[idx], size=8,
                     ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8),
                               ),
                     zorder=2)
            plt.plot(x, y, 'o', color='blue', ms=2, zorder=3)

        plt.show()
