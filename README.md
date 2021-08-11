# Deep Gabor Neural Network for Automatic Detection of Mine-like Objects in Sonar Imagery
## Descriptions
We address the automatic detection of mine-like objects using sonar images. The proposed Gabor-based detector is designed as a feature pyramid network
with a small number of trainable weights.

![alt_text](/output/REMUS100.png) ![alt_text](/output/test2.png) ![alt_text](/output/test3.png)

**Fig. 1.** The REMUS100 AUV used for sonar data acquisition. Figure courtesy of Maritime Survey Australia.

![alt_text](/output/sidescan_sonar.png) ![alt_text](/output/test2.png) ![alt_text](/output/test3.png)

**Fig. 2.** Principle of a side-scan sonar mounted on an autonomous underwater vehicle.

![alt_text](/output/test1.png) ![alt_text](/output/test2.png) ![alt_text](/output/test3.png)

**Fig. 2.** Representative visual results produced by the proposed Gabor-based method.

## Quick start
### Installation
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

## Citation
If you find this work or code is helpful for your research, please cite:
```
@ARTICLE{9095329,
  author={Thanh Le, Hoang and Phung, Son Lam and Chapple, Philip B. and Bouzerdoum, Abdesselam and Ritz, Christian H. and Tran, Le Chung},
  journal={IEEE Access}, 
  title={Deep Gabor Neural Network for Automatic Detection of Mine-Like Objects in Sonar Imagery}, 
  year={2020},
  volume={8},
  number={},
  pages={94126-94139},
  doi={10.1109/ACCESS.2020.2995390}}
  ```
## Reference
[1] H. T. Le, S. L. Phung, P. B. Chapple, A. Bouzerdoum, C. H. Ritz, and L. C. Tran, “Deep Gabor neural network for
automatic detection of mine-like objects in sonar imagery,” IEEE Access, vol. 8, pp. 94 126–94 139, 2020.
