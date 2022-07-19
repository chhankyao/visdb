# VisDB: Visibility-aware Dense Body



## Introduction

This repo contains the PyTorch implementation of "[Learning Visibility for Robust Dense Human Body Estimation]()" (ECCV'2022). Extended from a heatmap-based representation in [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE), we explicitly model the dense visibility of human joints and vertices to improve the robustness on partial-body images. 


## Setup
We implement VisDB with Python3 and PyTorch CUDA. Our code is mainly built upon this repo: [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE).

* Install **[PyTorch](https://pytorch.org)** and Python >= 3.7.3 and run `sh requirements.sh`. 
* Change `torchgeometry` kernel code slightly following [here](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527).
* Download the pre-trained VisDB models (available soon).
* Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl` and `basicModel_m_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](http://smplify.is.tue.mpg.de/). Place them under `common/utils/smplpytorch/smplpytorch/native/models/`.
* Download DensePose UV files `UV_Processed.mat` and `UV_symmetry_transforms.mat` from [here](https://dl.fbaipublicfiles.com/densepose/densepose_uv_data.tar.gz) and place them under `common/`.
* Download GMM prior `gmm_08.pkl` from [here](https://github.com/vchoutas/smplify-x/files/3295771/gmm_08.zip) and place it under `common/`.


## Quick demo
* Prepare `input.jpg` and pre-trained snapshot at `demo` folder.
* Go to `demo` folder and edit `bbox` in `demo.py`.
* run `python demo.py --gpu 0 --stage param --test_epoch 7` if you want to run on gpu 0.
* You can see output images in `demo`.
* If you run this code in ssh environment without display device, do follow:
```
1、Install oemesa follow https://pyrender.readthedocs.io/en/latest/install/
2、Reinstall the specific pyopengl fork: https://github.com/mmatl/pyopengl
3、Set opengl's backend to egl or osmesa via os.environ["PYOPENGL_PLATFORM"] = "egl"
```


## Contacts

Chun-Han Yao: <cyao6@ucmerced.edu>



## Citation

If you find our project useful in your research, please consider citing:

```
@inproceedings{yao2022learning,
  title={Learning visibility for robust dense human body estimation},
  author={Yao, Chun-Han and Yang, Jimei and Ceylan, Duygu and Zhou, Yi and Zhou, Yang, and Yang, Ming-Hsuan},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
}
```