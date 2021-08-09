import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from layer.gabor_layer_v4 import Gabor2D


def gabor_network(inputs, num_anchors, num_classes=1):
    """
    Create a Gabor Neural Network
    :param inputs: a pre-processed input sonar image as a 3D tensor
    :param num_anchors: number of anchors
    :param num_classes: number of classes (1 for MLO detection)
    :return: a Gabor Neural Network
    """
    # Branch 1
    gabor1 = Gabor2D(16, kernel_size=15, padding='same')(inputs)
    bn1 = BatchNormalization()(gabor1)
    relu1 = LeakyReLU(alpha=0.1)(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu1)

    gabor2 = Gabor2D(32, kernel_size=15, padding='same')(pool1)
    bn2 = BatchNormalization()(gabor2)
    relu2 = LeakyReLU(alpha=0.1)(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(relu2)

    gabor3 = Gabor2D(64, kernel_size=15, padding='same')(pool2)
    bn3 = BatchNormalization()(gabor3)
    x1 = LeakyReLU(alpha=0.1)(bn3)

    # Branch 2
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    gabor4 = Gabor2D(64, kernel_size=7, padding='same')(pool3)
    bn4 = BatchNormalization()(gabor4)
    relu4 = LeakyReLU(alpha=0.1)(bn4)

    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(relu4)
    gabor5 = Gabor2D(128, kernel_size=7, padding='same')(pool4)
    bn5 = BatchNormalization()(gabor5)
    relu5 = LeakyReLU(alpha=0.1)(bn5)

    gabor6 = Gabor2D(256, kernel_size=7, padding='same')(relu5)
    bn6 = BatchNormalization()(gabor6)
    x2 = LeakyReLU(alpha=0.1)(bn6)

    # Branch 3
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x2)
    gabor7 = Gabor2D(256, kernel_size=3, padding='same')(pool5)
    bn7 = BatchNormalization()(gabor7)
    relu7 = LeakyReLU(alpha=0.1)(bn7)

    pool6 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(relu7)
    gabor8 = Gabor2D(512, kernel_size=3, padding='same')(pool6)
    bn8 = BatchNormalization()(gabor8)
    relu8 = LeakyReLU(alpha=0.1)(bn8)

    gabor9 = Gabor2D(128, kernel_size=1, padding='same')(relu8)
    bn9 = BatchNormalization()(gabor9)
    x3 = LeakyReLU(alpha=0.1)(bn9)

    # Output branch 1
    gabor10 = Gabor2D(256, kernel_size=3, padding='same')(x3)
    bn10 = BatchNormalization()(gabor10)
    relu10 = LeakyReLU(alpha=0.1)(bn10)
    y1 = Gabor2D(num_anchors * (num_classes + 5), kernel_size=1, padding='same')(relu10)

    # Output branch 2
    gabor11 = Gabor2D(64, kernel_size=1, padding='same')(x3)
    bn11 = BatchNormalization()(gabor11)
    relu11 = LeakyReLU(alpha=0.1)(bn11)
    x4 = UpSampling2D(size=(2, 2))(relu11)

    x5 = Concatenate()([x4, x2])
    gabor12 = Gabor2D(128, kernel_size=3, padding='same')(x5)
    bn12 = BatchNormalization()(gabor12)
    relu12 = LeakyReLU(alpha=0.1)(bn12)
    y2 = Gabor2D(num_anchors * (num_classes + 5), kernel_size=1, padding='same')(relu12)

    # Output branch 3
    gabor13 = Gabor2D(32, kernel_size=1, padding='same')(x5)
    bn13 = BatchNormalization()(gabor13)
    relu13 = LeakyReLU(alpha=0.1)(bn13)
    x6 = UpSampling2D(size=(2, 2))(relu13)

    x7 = Concatenate()([x6, x1])
    gabor14 = Gabor2D(64, kernel_size=3, padding='same')(x7)
    bn14 = BatchNormalization()(gabor14)
    relu14 = LeakyReLU(alpha=0.1)(bn14)
    y3 = Gabor2D(num_anchors * (num_classes + 5), kernel_size=1, padding='same')(relu14)

    return Model(inputs, [y1, y2, y3])


def gnn_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """
    Convert final layer features to bounding box parameters
    :param feats: final features
    :param anchors: pre-defined anchors
    :param num_classes: number of classes
    :param input_shape: input shape
    :param calc_loss: option for calculating the loss
    :return:
    """
    num_anchors = len(anchors)

    # Reshape the features
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust predictions to each spatial grid point and anchor size
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def gnn_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def gnn_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = gnn_head(feats, anchors, num_classes, input_shape)
    boxes = gnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def gnn_eval(gnn_outputs, anchors, num_classes, model_shape,
             max_boxes=20, conf_threshold=0.5, iou_threshold=0.15):
    num_layers = len(gnn_outputs)  # 3 outputs for 3 scales
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.shape(gnn_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    # Get prediction boxes from 3 scales
    for l in range(num_layers):
        _boxes, _box_scores = gnn_boxes_and_scores(gnn_outputs[l], anchors[anchor_mask[l]],
                                                   num_classes, input_shape, model_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    # Remove weak predictions using a pre-defined threshold
    mask = box_scores >= conf_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    # Perform the "None-Maximum Suppression" algorithm to remove duplicate detections
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 16, 1: 8, 2: 4}[l] for l in range(num_layers)]
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue

        # Expand dim to apply broadcasting
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def box_iou(b1, b2):
    # Expand dim to apply broadcasting
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def gnn_loss(args, anchors, num_classes, iou_threshold=0.5, print_loss=False):
    num_layers = len(anchors) // 3

    # Get the network output and the corresponding ground-truth
    gnn_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(gnn_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(gnn_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(gnn_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(gnn_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        # Get the predicted bounding boxes
        grid, raw_pred, pred_xy, pred_wh = gnn_head(gnn_outputs[l], anchors[anchor_mask[l]], num_classes,
                                                    input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < iou_threshold, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss,
                                   class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
