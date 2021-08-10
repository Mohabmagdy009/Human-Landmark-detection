"""
Convolutional Neural Network for landmarks detection.
"""
import argparse
import tensorflow as tf
from tensorflow import keras
from dataset import make_dataset
from model import build_landmark_model
from config import *

# Introduce arguments parser to give user the flexibility to tune the process.
parser = argparse.ArgumentParser()
parser.add_argument('--train_set', default='data/train.txt', type=str,
                    help='Training dataset')
parser.add_argument('--val_set', default='data/valid.txt', type=str,
                    help='validation dataset')
parser.add_argument('--epochs', default=100, type=int,
                    help='epochs for training')
parser.add_argument('--batch_size', default=64, type=int,
                    help='training batch size')
parser.add_argument('--export_only', default=False, type=bool,
                    help='Save the model without training and evaluation.')
parser.add_argument('--eval_only', default=False, type=bool,
                    help='Do evaluation without training.')
args = parser.parse_args()


def save_model(model):
    if not tf.io.gfile.exists(export_dir):
        tf.io.gfile.mkdir(export_dir)

    print("Saving model to {} ...".format(export_dir))
    model.save(export_dir, include_optimizer=False)
    print("Model saved at: {}".format(export_dir))    


if __name__ == '__main__':
    # Create the Model
    model = build_landmark_model(input_shape=input_shape,
                                 output_size=num_marks*2)

    # Prepare for training. First restore the model if any checkpoint file available.
    if not tf.io.gfile.exists(checkpoint_dir):
        tf.io.gfile.mkdir(checkpoint_dir)
        print("Checkpoint directory created: {}".format(checkpoint_dir))

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Checkpoint found: {}, restoring...".format(latest_checkpoint))
        model.load_weights(latest_checkpoint)
        print("Checkpoint restored: {}".format(latest_checkpoint))
    else:
        print("Checkpoint not found. Model weights will be initialized randomly.")

    # Sometimes the user only want to save the model. Skip training in this case.
    if args.export_only:
        save_model(model)
        quit()

    # Finally, it's time to train the model.
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss=keras.losses.mean_squared_error)

    # Construct a dataset for evaluation.
    dataset_val = make_dataset(dataset_file=args.val_set,
                               batch_size=args.batch_size,
                               shuffle=False)

    # If evaluation is required only.
    if args.eval_only:
        print('Starting to evaluate.')
        evaluation = model.evaluate(dataset_val)
        print(evaluation)
        quit()

    # To save and log the training process, we need some callbacks.
    callback_tb = keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=10)
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir+"/landmark",
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        verbose=1)
    callbacks = [callback_tb, callback_checkpoint]

    # Get the training data ready.
    dataset_train = make_dataset(dataset_file=args.train_set,
                                 batch_size=args.batch_size,
                                 shuffle=True)

    model.fit(dataset_train, validation_data=dataset_val, epochs=args.epochs,
              callbacks=callbacks)

    save_model(model)
