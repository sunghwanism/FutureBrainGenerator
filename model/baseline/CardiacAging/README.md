# Cardiac aging synthesis from cross-sectional data with conditional generative adversarial networks

## Requirements
 pip install pytorch_lightning==1.4.7 scikit-learn==1.6.1 nibabel==3.2.1 monai==0.8.0 tensorboard==2.18.0 pandas==2.2.3 ffmpeg-python==0.2.0 grad-cam==1.5.4 comet-ml==3.47.6 munch==2.5.0 pillow==7.0.0
- python=3.9.21
- pytorch_lightning=1.4.7
- scikit-learn=1.6.1
- nibabel=3.2.1
- monai=0.8.0
- tensorboard=2.18.0
- pandas=2.2.3
- ffmpeg-python=0.2.0
- grad-cam=1.5.4
- comet-ml=3.47.6
- munch=2.5.0
- pillow
- torchmetrics=0.6.0
- wandb
- numpy 1.23.0

### Fix MONAI to avoid error! - Optional(if error)
monai.utilis.misc.py -> line 52 : "MAX_SEED = np.iinfo(np.uint32).max +1" to "MAX_SEED = np.iinfo(np.uint32).max - 1"
