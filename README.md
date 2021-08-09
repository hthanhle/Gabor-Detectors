# Deep Gabor Neural Network for Automatic Detection of Mine-like Objects in Sonar Imagery
## Quick start
### Install
1. Install Tensorflow=1.13.1 and Keras=2.2.4 following [the official instructions](https://www.tensorflow.org/install/pip)

2. git clone https://github.com/hthanhle/Gabor-Detectors/

3. Install dependencies: `pip install -r requirements.txt`

### Train and test

Please specify the configuration file. 

1. To train a Gabor Detector, run the following command:

**Example 1:** `python train.py -- dataset ./data/folds/1/train.text`

**Example 2:** `python train.py -- dataset ./data/folds/1/train.text --batch 16`

2. To test a pretrained Gabor Detector, run the following command:

**Example 1:** `python test.py --input images/Training00001.bmp --output images/Training00001-output.jpg`

**Example 2:** `python test.py --input images/Training00001.bmp --output images/Training00001-output.jpg --conf 0.3 --iou 0.2`

**Example 3:** `python test.py --input images/Training00001.bmp --output images/Training00001-output.jpg --weight ./checkpoints/ep157-loss5.182-val_loss4.437.h5 --conf 0.3 --iou 0.2`


