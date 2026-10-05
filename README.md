# Automated bitewing chart filing

Comprehensive bitewing chart filing using hierarchical instance segmentation.

This code is implemented using OneDL-MMDetection (v.3.5.1) based on PyTorch 2.10.0.


## Installation

See `INSTALL.md` for installation instructions.


## Data and checkpoints

The panoramic radiographs from the OdontoAI platform can be requested from [the platform's website](https://odontoai.com/). Furthermore, model checkpoints and the data collected for the study can be requested at the [Radboud Data Repository](https://doi.org/10.34973/1bf3-j248).


## Paths

Data, checkpoint and output locations are not hard-coded. Every config file and script reads them from these environment variables, and every script also accepts them as command-line options (`--data-root`, `--work-dirs`, `--odonto-root`):

| Variable | Contents | Default |
|---|---|---|
| `BITEWINGS_DATA_ROOT` | Dataset folder with `images/`, `annotations.json` and (after preprocessing) `splits/` | `../data/Netherlands` |
| `BITEWINGS_CHECKPOINTS` | Folder with the `*.pth` checkpoints | `../checkpoints` |
| `BITEWINGS_WORK_DIRS` | Folder for training runs and predictions (`chart_filing_<model>/`, `odonto_bitewings_<model>/`) | `work_dirs` |
| `BITEWINGS_ODONTO_ROOT` | OdontoAI data for pre-training (output goes to `<root>/bitewings`) | `../data/odontoai` |

The defaults reproduce the original layout, so the commands below work unchanged when the data sits next to the repository. Config files and scripts can be run from any working directory; `PYTHONPATH=.` is no longer required. For example:

```bash
export BITEWINGS_DATA_ROOT=/mnt/data/Netherlands
export BITEWINGS_CHECKPOINTS=/mnt/checkpoints
export BITEWINGS_WORK_DIRS=/mnt/runs

python bitewings/preprocess/preprocess.py
python onedl-mmdetection/tools/train.py bitewings/configs/config_hierarchical.py
```


## Docker

Two images are provided; build either from the repository root with the submodule checked out:

```bash
docker build -f docker/Dockerfile.cpu -t bitewings:cpu .   # inference, preprocessing, smoke tests
docker build -f docker/Dockerfile.gpu -t bitewings:gpu .   # training (CUDA 12.9, ~20-40 min build)
```

Both set `BITEWINGS_DATA_ROOT=/data`, `BITEWINGS_CHECKPOINTS=/checkpoints` and `BITEWINGS_WORK_DIRS=/work_dirs`, so mount those folders:

```bash
docker run --rm --gpus all -v <data>:/data -v <checkpoints>:/checkpoints -v <runs>:/work_dirs bitewings:gpu \
  python onedl-mmdetection/tools/train.py bitewings/configs/config_finetune.py
```


## Inference

The simplest way to run the hierarchical model on a folder of bitewings is `config_inference.py`:

```bash
python bitewings/inference/init_images.py <images>
BITEWINGS_IMAGES=<images> python onedl-mmdetection/tools/test.py \
  bitewings/configs/config_inference.py <checkpoints>/hierarchical_chartfiling.pth
```

Predictions are written to `<images>/detections.pkl`. By default it uses low-memory post-processing (`BITEWINGS_LOW_MEMORY=1`): candidates with a class probability below 0.1 are dropped before full-resolution masks are built and each tooth keeps one FDI label, so CPU inference fits in about 3 GB of RAM. Detections scoring 0.1 or higher are unchanged; set `BITEWINGS_LOW_MEMORY=0` to reproduce the original post-processing exactly.

Alternatively, to run the model on your own bitewings with any architecture, first make an empty COCO file by running `bitewings/inference/init_images.py`. The model can be run on these images using this command:

```bash
export IN_DIR=`realpath "<path>"`
python \
  onedl-mmdetection/tools/test.py \
  bitewings/configs/config_<model>.py \
  checkpoints/<model>_chartfiling.pth \
  --cfg-options \
    test_dataloader.dataset.data_root="$IN_DIR" \
    test_dataloader.dataset.data_prefix.img="$IN_DIR" \
    test_dataloader.dataset.ann_file="$IN_DIR/coco.json"
```

choosing the input directory for `<path>` and a model architecture for `<model>`. The predictions can be converted to COCO by running `bitewings/inference/mmdet2coco.py`. Afterward, the annotations can be visualized by running `bitewings/inference/show_anns.py` or `bitewings/visualization/side_by_side.py`. One example bitewing has been added in the `test/` folder with which you can run the models and show the prediction results.


## OdontoAI pre-training

If you would like to skip this step, model checkpoints pre-trained on COCO and OdontoAI are made available on request.

**Preprocessing** Unzip the downloaded data from the OdontoAI platform to the `../data/odontoai` folder (or set `BITEWINGS_ODONTO_ROOT`) and run `bitewings/odonto/preprocess.py` to combine the train and validation images and to split the data.

**Training** Following the preprocessing, the Mask DINO, Mask R-CNN, and SparseInst models can be pre-trained by running the following command:

```bash
PYTHONPATH=. python onedl-mmdetection/tools/train.py bitewings/odonto/config_<model>.py
```

choosing a model architecture for `<model>`. The training run will be logged using TensorBoard and the checkpoints and logging files will be stored in `work_dirs/odonto_bitewings_<model>`.


## Fine-tuning

### Preprocessing

Unzip the downloaded data collected from The Netherlands to the `../data/Netherlands` folder (or set `BITEWINGS_DATA_ROOT`). As the teeth and tooth findings were annotated independently, each tooth finding needs to be matched to a tooth with corresponding FDI number. Please run `bitewings/preprocess/preprocess.py` to automatically assign tooth findings to teeth and to split the images into train, validation, and test.

Furthermore, `bitewings/preprocess/intensities.py` and `bitewings/preprocess/prevalences.py` can be run to visualize the intensity distribution and to show the tooth finding prevalances of the bitewings from The Netherlands. Please note that the data from The Netherlands does not include implants.


### Training

After pre-training a model on the OdontoAI dataset, the model checkpoint in the working directory can be copied to `../checkpoints/<model>_odonto.pth`, after which it can be fine-tuned on the bitewings from The Netherlands using the following command:

```bash
PYTHONPATH=. python onedl-mmdetection/tools/train.py bitewings/configs/config_<model>.py
```

choosing a model architecture for `<model>`. Please note that the hierarchical instance segmentation method requires at least 20 GPU hours to complete training.

### Fine-tuning the released model on new data

To add data to the already-trained model rather than reproduce the paper, use `config_finetune.py`. It starts from `hierarchical_chartfiling.pth` and trains for 12 epochs at a learning rate of 2e-5 (10x lower for the last quarter):

```bash
BITEWINGS_TRAIN_ANN=splits/train_fdi_1.json BITEWINGS_VAL_ANN=splits/val_fdi_1.json \
  python onedl-mmdetection/tools/train.py bitewings/configs/config_finetune.py
```

### Adding front teeth from DENTEX panoramics

The Radboud bitewings contain almost no incisors, so the released model rarely finds front teeth on periapicals. [DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) (MICCAI 2023, **CC BY-NC-SA 4.0, non-commercial**) numbers every tooth of 634 panoramics. `config_finetune_dentex.py` trains on the Radboud bitewings plus periapical-style crops of these panoramics (`CropPeriapical`, front teeth included). DENTEX has no finding labels, so its images are flagged with `MarkPartialFindings` and only teach tooth outlines and numbers; findings are still learned from Radboud.

```bash
# once: convert DENTEX's quadrant_enumeration set and split it 80/10/10
python bitewings/dentex/convert.py --dentex-root <DENTEX>/training_data/quadrant_enumeration
# optional: look at what training will see
python bitewings/dentex/preview_crops.py --dentex-root <...>/quadrant_enumeration --out-dir crop_preview

BITEWINGS_DENTEX_ROOT=<...>/quadrant_enumeration \
  python onedl-mmdetection/tools/train.py bitewings/configs/config_finetune_dentex.py
```

Models trained with DENTEX inherit its non-commercial license.

The annotation files must be in the hierarchical (`*_fdi_*.json`) format produced by `bitewings/preprocess/preprocess.py`. Further settings (`BITEWINGS_EPOCHS`, `BITEWINGS_LR`, `BITEWINGS_BATCH_SIZE`, `BITEWINGS_NUM_WORKERS`, `BITEWINGS_RUN_NAME`, `BITEWINGS_INIT_CHECKPOINT`) are documented at the top of the config. The best checkpoint by aggregate finding F1 and the latest checkpoint are kept in `$BITEWINGS_WORK_DIRS/<run name>`.


### Evaluation

After fine-tuning a model on the data from The Netherlands, the model checkpoint in the working directory can be copied to `../checkpoints/<model>_chartfiling.pth`. A fine-tuned model can be evaluated using the following command:

```bash
PYTHONPATH=. python onedl-mmdetection/tools/test.py bitewings/configs/config_<model>.py ../checkpoints/<model>_chartfiling.pth
```

choosing a model architecture for `<model>`. This will produce mean average precision (mAP) metrics for tooth segmentation and labeling and the results will be written to a pickle file in the working directory.

Further metrics for tooth segmentation and labeling can be computed by running `bitewings/evaluation/tooth_segmentation.py`. Furthermore, metrics can be computed for tooth finding classification by running `bitewings/evaluation/tooth_findings.py`. Lastly, figures showing the original bitewing, the bitewing with annotations, and the bitewing with model predictions can be visualized by running `bitewings/visualization/side_by_side.py`.


## Citation

```
@article{CAO2025105919,
  title = {Automated Chart Filing on Bitewings Using Deep Learning: Enhancing Clinical Diagnosis in a Multi-Center Study},
  journal = {Journal of Dentistry},
  pages = {105919},
  year = {2025},
  doi = {https://doi.org/10.1016/j.jdent.2025.105919},
  author = {Lingyun Cao and Niels {van Nistelrooij} and Eduardo Trota Chaves and Stefaan Bergé and Maximiliano Sergio Cenci and Tong Xi and Bas Loomans and Shankeeth Vinayahalingam}
}
```
