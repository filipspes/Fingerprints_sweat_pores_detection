## <div align="center">Introduction</div>
This project was created to detect sweat pores on fingerprints images using One-stage detector YOLOv5 (Official repository: https://github.com/ultralytics/yolov5) and the Two-stage detector Mask R-CNN (Official repository: https://github.com/matterport/Mask_RCNN).

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

Create Ancaonda environment with Python 3.8 and clone this repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.8.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
conda create --name=<name of environment> python=3.8
conda activate <name of environment>
git clone https://github.com/filipspes/Fingerprints_sweat_pores_detection  # clone
cd Fingerprints\_sweat\_pores\_detection
cd GUI
pip install -r requirements.txt  # install
python gui.py # run
```
