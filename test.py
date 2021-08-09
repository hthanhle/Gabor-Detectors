import argparse
import numpy as np
import keras.backend as K
from keras.layers import Input
from model.model import gnn_eval, gabor_network
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
from utils.utils import resize_image, get_anchors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


def detect_image(model, in_file, out_file, image_size, model_shape, boxes, scores, classes, font):
    """
    Detect MLOs in a single sonar image
    :param model: trained Gabor Detector
    :param in_file: path of the input image
    :param out_file: path of the output image
    :param image_size: image size
    :param model_shape: expected model shape
    :param boxes: boxes
    :param scores: scores
    :param classes: classes
    :param font: font
    """
    # Read the input image
    image = Image.open(in_file)
    image = image.convert('RGB')

    # Pre-process the image
    resized_image = resize_image(image, tuple(reversed(image_size)))
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)

    # Run the session with the actual values passed to the placeholders
    start = timer()
    sess = K.get_session()
    out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],
                                                  feed_dict={model.input: image_data,
                                                             model_shape: [image.size[1], image.size[0]],
                                                             K.learning_phase(): 0
                                                             })
    end = timer()
    print('Found {} objects'.format(len(out_boxes)))
    print('Inference time: {}(s)'.format(end - start))

    # Visualize all predicted bounding boxes (i.e., detected MLOs)
    for i, c in reversed(list(enumerate(out_classes))):

        # Get a predicted box
        box = out_boxes[i]
        score = out_scores[i]

        # Visualize the result
        label = '{:.2f}'.format(score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        # Get the location of the bounding box
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print('Object type: {}, Score: {:.2f}, Location: '.format('MLO', score), [left, top], [right, bottom])

        # Adjust the bounding box if it is outside the input image (just for a better visualization)
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1] - 3])  # 3: the gap between the bounding box and the label
        else:
            text_origin = np.array([left, top + 3])

        # Draw a rectangle covering the detected MLO
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 0, 0), width=2)  # box in red color

        # Draw an outside rectangle covering the text
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=(255, 0, 0))

        # Draw the text
        draw.text(text_origin, label, fill=(255, 255, 255), font=font)  # text in white color

        # Delete the current draw object
        del draw

    image.save(out_file)
    image.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Test a trained Deep Gabor Network for MLO detection')
    args.add_argument('-i', '--input', default=None, type=str, help='Path of input image file')
    args.add_argument('-s', '--size', default=832, type=int, help='Image size')
    args.add_argument('-a', '--anchor', default='./data/anchors.txt', type=str, help='Path of the anchor file')
    args.add_argument('-w', '--weight', default='./checkpoints/ep157-loss5.182-val_loss4.437.h5', type=str,
                      help='Path of the trained weights')
    args.add_argument('-c', '--conf', default=0.5, type=float, help='Confidence threshold for removing weak detections')
    args.add_argument('-u', '--iou', default=0.15, type=float, help='IOU threshold for removing duplicate detections')
    cmd_args = args.parse_args()
    assert cmd_args.input is not None, "Path of input image file is missing"
    assert cmd_args.size % 32 == 0, 'Image size must be a multiple of 32'

    # Get the anchors from file
    anchors = get_anchors(cmd_args.anchor)

    # Create a new GNN, and then load the trained weights
    model = gabor_network(Input(shape=(None, None, 3)),
                          num_anchors=len(anchors) // 3,
                          num_classes=1)

    model.load_weights(cmd_args.weight)
    print('Loading the trained Gabor network ...')

    # Evaluate the model by creating an instance with placeholders
    model_shape = K.placeholder(shape=(2,))
    boxes, scores, classes = gnn_eval(model.output, anchors,
                                      num_classes=1,
                                      model_shape=model_shape,
                                      conf_threshold=cmd_args.conf,
                                      iou_threshold=cmd_args.iou)

    # Perform detection with the actual inputs
    detect_image(model,
                 in_file=cmd_args.input,
                 out_file=cmd_args.input[:-4] + '-out.jpg',
                 image_size=(cmd_args.size, cmd_args.size),
                 model_shape=model_shape,
                 boxes=boxes,
                 scores=scores,
                 classes=classes,
                 font=ImageFont.truetype('./utils/FiraMono-Medium.otf', size=25))
