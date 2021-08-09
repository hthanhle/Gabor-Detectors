import keras.backend as K
import numpy as np
import argparse
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from model.model import gabor_network, gnn_loss, preprocess_true_boxes
from utils.utils import get_random_data, get_anchors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


def create_model(anchors, img_size=(832, 832), num_classes=1):
    """
    Create a Gabor detector
    :param anchors: pre-defined anchors
    :param img_size: image size
    :param num_classes: number of classes (1 for MLO detection)
    :return: a Gabor Neural Network
    """
    K.clear_session()
    input_img = Input(shape=(None, None, 3))
    h, w = img_size
    num_anchors = len(anchors)

    # Specify the expected output shapes corresponding to the three scales (16, 8, 4)
    y_true = [Input(shape=(h // {0: 16, 1: 8, 2: 4}[l], w // {0: 16, 1: 8, 2: 4}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]

    # Create a Gabor network (i.e., model body)
    model_body = gabor_network(input_img, num_anchors // 3, num_classes)

    # Define the loss (actual outputs vs. y_true)
    model_loss = Lambda(gnn_loss, output_shape=(1,), name='gnn_loss',
                        arguments={'anchors': anchors,
                                   'num_classes': num_classes,
                                   'iou_threshold': 0.7})([*model_body.output, *y_true])

    # Build the entire Gabor detector from the defined loss and model body
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_list, anchors, batch_size=8, img_size=(832, 832), num_classes=1):
    """
    Create a data generator
    :param annotation_list: a annotation list (i.e., ground-truths)
    :param anchors: pre-defined anchors
    :param batch_size: batch size
    :param img_size: image size
    :param num_classes: number of classes
    :return: a data generator
    """
    n = len(annotation_list)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_list)
            image, box = get_random_data(annotation_list[i], img_size, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, img_size, anchors, num_classes)

        # Return a generator
        yield [image_data, *y_true], np.zeros(batch_size)


if __name__ == '__main__':

    # Parse the command arguments
    args = argparse.ArgumentParser(description='Train a Deep Gabor Network for MLO detection')
    args.add_argument('-d', '--dataset', default=None, type=str, help='Path of the training annotation list')
    args.add_argument('-s', '--size', default=832, type=int, help='Image size')
    args.add_argument('-b', '--batch', default=8, type=int, help='Batch size')
    args.add_argument('-a', '--anchor', default='./data/anchors.txt', type=str, help='File path of the anchor file')
    cmd_args = args.parse_args()
    assert cmd_args.dataset is not None, "Path of the training annotation list is missing"
    assert cmd_args.size % 32 == 0, 'Image size must be a multiple of 32'

    # Get the anchors from file
    anchors = get_anchors(cmd_args.anchor)

    # Create a Gabor model
    model = create_model(anchors=anchors,
                         img_size=(cmd_args.size, cmd_args.size),
                         num_classes=1)
    model.summary()

    # Define callbacks for training
    early_stop = EarlyStopping(monitor='val_loss', patience=20)
    checkpoint = ModelCheckpoint('./checkpoints/ep{epoch:03d}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 period=1)
    callbacks = [early_stop, checkpoint]

    # Read the training text file
    with open(cmd_args.dataset) as f:
        train_dataset = f.readlines()

    # Create training/valid data generators from the annotation list
    num_train = round(len(train_dataset) * 0.9)  # reserve 10% for validation
    num_val = len(train_dataset) - num_train

    train_gen = data_generator(annotation_list=train_dataset[:num_train],
                               anchors=anchors,
                               batch_size=cmd_args.batch,
                               img_size=(cmd_args.size, cmd_args.size),
                               num_classes=1)
                               
    val_gen = data_generator(annotation_list=train_dataset[num_train:],
                             anchors=anchors,
                             batch_size=cmd_args.batch,
                             img_size=(cmd_args.size, cmd_args.size),
                             num_classes=1)

    # Compile the model
    model.compile(optimizer=Adam(lr=1e-3),
                  loss={'gnn_loss': lambda y_true, y_pred: y_pred})

    # Fit the model
    model.fit_generator(generator=train_gen,
                        validation_data=val_gen,
                        epochs=200,
                        steps_per_epoch=max(1, num_train // cmd_args.batch),
                        validation_steps=max(1, num_val // cmd_args.batch),
                        callbacks=callbacks)
