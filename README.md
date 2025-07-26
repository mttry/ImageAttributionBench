# ImageAttributionBench
This is the offical repository for "ImageAttributionBench: How Far Are We from Semantic-Free Attribution?"
## Dependencies
You can create the environment using the provided `environment.yaml` file:
```
git clone git@github.com:mttry/ImageAttributionBench.git
cd ImageAttributionBench
conda env create -f environment.yaml


```
## 🚆 Dataset Construction Pipeline

The dataset construction process is organized under the `dataset_construction` directory, which contains all necessary components for generating high-quality image-text datasets.  
For detailed usage and module descriptions, please refer to [`dataset_construction/README.md`](./dataset_construction/README.md).

## Dataset
The dataset **ImageAttributionBench** is available [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O4S4IV). After downloading, you will obtain a dataset with the following structure:
```
/home/image_attribution_dataset
|-- 4o
|-- CogView3_PLUS
...
`-- real

/home/image_attribution_dataset/4o/
|-- AnimalFace
|   |-- cat
|   |-- dog
|   `-- wild
|-- COCO
|-- HumanFace
|   |-- FFHQ
|   `-- celebahq
|-- ImageNet-1k
`-- Scene
    |-- bedroom
    |-- church
    `-- classroom
```
Modify the default value of the root_dir argument in both `training/train.py` and `training/test.py` to point to the location where you downloaded the dataset.

📌 Update
We have added a new script: `dataset/download.py` that allows you to automatically download selected dataset files from Dataverse.
After filling in your Harvard Dataverse API Token, you can use the script to download specific models and semantic classes as follows:
```
python dataset/download.py \
  --OUTPUT_DIR "./downloaded_data" \
  --MODEL_CLASSES "mid-6.0,4o,SD1_5" \
  --SEMANTIC_CLASSES "COCO,FFHQ,dog" \
  --DELETE_ZIP True
```

or download the full dataset:
```
python dataset/download.py \
  --OUTPUT_DIR "./downloaded_data" \
  --DELETE_ZIP True
```

## Weights 
you can download trained weights of attributors at [Harvard DataVerse](https://doi.org/10.7910/DVN/7IEAXP).Place weights in `training/ckpt` like 
```
training/ckpt
|-- dct
|   |-- dct_default.pth
|   |-- dct_split1.pth
|   |-- dct_split2.pth
|   `-- dct_split3.pth
|-- defl
|   |-- defl_default.pth
|   |-- defl_split1.pth
|   |-- defl_split2.pth
|   `-- defl_split3.pth
...
`-- ucf
    |-- ucf_default.pth
    |-- ucf_split1.pth
    |-- ucf_split2.pth
    `-- ucf_split3.pth
```

## Train and evaluate
You can use the scripts in `training/scripts` for training and `training/scripts_test` for testing. Take `ResNet50` for example:
```
cd training

# training:
bash scripts/resnet50.bash

# testing:
bash scripts/resnet50.bash
```
or you can use `training/train.py` and `training/test.py`:
```
cd training

# training:
# standart-split task:
python train.py --config config/model/resnet50.yaml  --n_epoch 10
# semantic task:
python train.py --use_semantic_split --task_id 1 --config config/model/resnet50.yaml  --n_epoch 10

# testing:
# standart-split task:
python test.py  --config config/model/resnet50.yaml  --resume_checkpoint ckpt/resnet50/resnet50_default.pth
# semantic task:
python test.py  --use_semantic_split --task_id 1 --config config/model/resnet50.yaml --resume_checkpoint ckpt/resnet50/resnet50_split1.pth
```

## Reference
