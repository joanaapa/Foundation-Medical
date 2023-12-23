# Are Natural Domain Foundation Models Useful for Medical Image Classification?

Codebase for the paper [*"Are Natural Domain Foundation Models Useful for Medical Image Classification?"*](https://arxiv.org/abs/2310.19522). \
_**Originally published at the Winter Conference on Applications of Computer Vision (WACV) 2024**_

<p align="center">
  <img width="45%" src="images/overall_comparison.png">
</p>


#### Abstract:
The deep learning field is converging towards the use of general foundation models that can be easily adapted for diverse tasks. While this paradigm shift has become common practice within the field of natural language processing, progress has been slower in computer vision. In this paper we attempt to address this issue by investigating the transferability of various state-of-the-art foundation models to medical image classification tasks. Specifically, we evaluate the performance of five foundation models, namely SAM, SEEM, DINOv2, BLIP, and OPENCLIP across four well-established medical imaging datasets. We explore different training settings to fully harness the potential of these models. Our study shows mixed results. DINOv2 consistently outperforms the standard practice of IMAGENET pretraining. However, other foundation models failed to consistently beat this established baseline indicating limitations in their transferability to medical image classification tasks


## Enviroment setup
To create a conda environment use the following command:\
```conda env create -f env_files/environment.yml``` 

To use Clip, it is required to install the open_clip package in editor mode:\
```pip install -e open_clip``` \
and replace the files ```model.py``` and  ```transformer.py``` with the ones found in [this folder](./env_files/open_clip).


## User guide

### Parameters
The main parameters that need to be specified are the following.

| Parameter           | Description                            |
|---------------------|----------------------------------------|
| dataset_params      | Dataset options are: APTOS2019, DDSM, CheXpert or ISIC2019. Data location is expected to be the folder containing       all of these datasets. |
| dataloader_params   | Specifications of the different datalodaders including batch size. |
| model_params        | Model used as a classifier, either "deit_base" or "none" for a linear classifier.  The pretraining type of the transformer can be either "supervised" or "dino". |
| foundation_params   | Foundation model used: "sam_vit_b", "dinov2_vitb14", "focal" (SEEM), "clip_vitb16", "blip_base" or "none". Includes the option of unfreezing or dropping the last layers of the model. |
| optimization_params | Mix of parameters used for optimization. By default a linear warmup is used.         |
| training_params     | Training specifications e.g. number of epochs, log frequency,... |
| log_params          | Parameters to log the training using WANDB.  |

### Usage
All the results can be reproduced by running the [main file](./classification.py) with the desired set of parameters specified in the config file.

Training example:\
```python classification.py --params_path "./params.json"``` 

A flag can be added for testing:\
```python classification.py --params_path "./params.json" --test```


## Citation
To cite our work please use the following BibTeX entry:

```markdown
@inproceedings{huix2024natural,
  title={Are Natural Domain Foundation Models Useful for Medical Image Classification?},
  author={Huix, Joana Pal{\'e}s and Ganeshan, Adithya Raju and Haslum, Johan Fredin and S{\"o}derberg, Magnus and Matsoukas, Christos and Smith, Kevin},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={7634--7643},
  year={2024}
}
```
