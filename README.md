# Human-Landmark-detection
### Human Landmark Detection

## Keras sequential CNN model.
    # input: 128x128 3-channel image
    # output: 24 floating point numbers each of 2 represents one landmark position.
    # 12 landmarks points are as follow:
        "L.Chest", "R.Chest", "Shoulder.A", "Shoulder.B", "Shoulder.C", "Shoulder.D", "Shoulder.E", "Arm.A", "Arm.B", "L.Waist", "R.Waist", "Arm.E"
    # model architecture
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    image_input (InputLayer)     [(None, 128, 128, 3)]     0
    _________________________________________________________________
    normalization (Normalization (None, 128, 128, 3)       7
    _________________________________________________________________
    conv2d (Conv2D)              (None, 126, 126, 32)      896
    _________________________________________________________________
    batch_normalization (BatchNo (None, 126, 126, 32)      128
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 63, 63, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 61, 61, 64)        18496
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 61, 61, 64)        256       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 59, 59, 64)        36928
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 59, 59, 64)        256
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 29, 29, 64)        0
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 27, 27, 64)        36928
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 27, 27, 64)        256
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 25, 25, 64)        36928
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 25, 25, 64)        256
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 12, 12, 64)        0
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 10, 10, 128)       73856
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 10, 10, 128)       512
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 8, 8, 128)         147584
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 8, 8, 128)         512
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 128)         0
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 5, 5, 256)         295168
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 5, 5, 256)         1024
    _________________________________________________________________
    flatten (Flatten)            (None, 6400)              0
    _________________________________________________________________
    dense (Dense)                (None, 1024)              6554624
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 1024)              4096
    _________________________________________________________________
    dense_1 (Dense)              (None, 24)                24600
    =================================================================
    Total params: 7,233,311
    Trainable params: 7,229,656
    Non-trainable params: 3,655
    _________________________________________________________________

## environment.
    - OS: windows, Linux, MacOs
    - python = 3.8

## set up environment.
    # install python = 3.8
    # install dependencies in requirements.txt
        $ pip install -r requirements.txt

## dataset preparation.
    # put your image dataset into anywhere.
    # prepare manifest files for training (train.txt, valid.txt, test.txt).
    # train.txt is for train, valid.txt is for validation while training, test.txt is for test.
    # Every line in manifest file contains "image path" and landmarks as following:
    # "image path", 24 floating points numbers which represents absolute positions of 12 landmark points in the image
        ex. data/auged/train_a1_1heavy_0001.jpg,4.33E+01,5.92E+01,1.34E+02,5.64E+01,3.21E+01,3.50E+01,5.54E+01,2.57E+01,8.47E+01,1.63E+01,1.24E+02,2.67E+01,1.39E+02,3.47E+01,2.35E+01,7.66E+01,4.87E+01,8.17E+01,5.63E+01,1.27E+02,1.28E+02,1.21E+02,1.47E+02,1.57E+02
    # All the items of a line must be separated by comma.

    # To preprocess DeepFashion dataset to train landmark detection model with this project
        $ manifestDeepFashion.py -d "Deepfashion Dataset extracted path" -o "output path to write manifest files"
        After create these manifest files we can train the model with landmark.py script.

## train model.
    # prepare configuration.
        Adjust config.py file to set up proper training parameters.
        All the default parameters are good for current dataset, but you can change something for other dataset.
    # train the model with landmark.py
        $ python landmark.py --train_set "path to train.txt" --val_set "path to valid.txt" --epochs "number of training epoch" --batch_size "batch size of each iteration" --export_only [True/False] --eval_only [True/False]

## inference.
    $ python inference.py
    you can input any image file path then it will show the predicted output with blue dots and proper annotations.
