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


## Inference

To run the model on your own bitewings, first make an empty COCO file by running `bitewings/inference/init_images.py`. The model can be run on these images using this command:

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
