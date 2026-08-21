# FH-Net

This repository contains the code release for **FH-Net: Frequency-Domain and Hierarchical Dual Enhancement for Few-Shot Surface Defect Segmentation**. The release retains the paper's ResNet-50 FH-Net path and provides reproducible configurations for the three FSSD-12 folds under 1-shot and 5-shot settings.

FH-Net uses two principal components:

- **SPFP** enhances support foreground prototypes through stochastic spatial and frequency-domain perturbations.
- **HCAM/SCPP** aggregates local, scale-aware, and global context before hierarchical prediction fusion.

Dataset images, pretrained backbone weights, trained checkpoints, and experiment outputs are not included.

## Repository layout

```text
FH-Net-release/
├── config/FSSD-12/{1shot,5shot}/
├── data_list/FSSD-12/{train,val}/
├── model/                 # FH-Net, SPFP, HCAM/SCPP, and ResNet-50
├── scripts/smoke_test.py
├── util/                  # dataset, transforms, configuration, metrics
├── train.py
└── test.py
```

## Environment

The exact package versions used for the original experiments were not recorded in the supplied source folder, so this release does not invent version pins. Create an isolated Python 3 environment, install a PyTorch build suitable for your CUDA runtime, and then install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the CPU-only construction and forward-pass check with:

```bash
python scripts/smoke_test.py
```

## FSSD-12 preparation

Place FSSD-12 under `datasets/FSSD-12`. Each sample path in the supplied fold lists points to an image under an `Images/` directory. The loader derives its mask path by replacing `Images/` with `GT/` and changing the `.jpg` suffix to `.png`.

```text
datasets/FSSD-12/
├── <defect-class>/
│   ├── Images/
│   │   └── <sample>.jpg
│   └── GT/
│       └── <sample>.png
└── ...
```

The six retained lists under `data_list/FSSD-12/` preserve the original sample paths, class identifiers, and fold assignments. FSSD-12 uses three folds; each fold holds out four of the twelve classes for evaluation, as implemented in `util/dataset.py`. Report results across all three folds for each shot setting.

No verified Surface Defects-4i configuration or data list was present in the supplied working folder, so none has been generated.

## ResNet-50 pretrained weights

Training initializes the paper's custom deep-base ResNet-50 from `initmodel/resnet50_v2.pth`. Create the directory and place the compatible file there:

```text
FH-Net-release/initmodel/resnet50_v2.pth
```

Alternatively, set `FHNET_RESNET50_WEIGHTS` to the local path of the same compatible weight file. The original source did not record a verifiable download source, so this repository does not provide or invent a download link. Testing a complete FH-Net checkpoint does not require this separate backbone file.

## Training

Run commands from the repository root. Example 1-shot and 5-shot fold-0 runs are:

```bash
python train.py --config config/FSSD-12/1shot/fold0_train.yaml
python train.py --config config/FSSD-12/5shot/fold0_train.yaml
```

Use `fold1_train.yaml` and `fold2_train.yaml` for the other folds. The supplied experimental hyperparameters are preserved. Checkpoints and TensorBoard events are written below `exp/FSSD-12/`, as specified by each training configuration.

## Evaluation

Place trained checkpoints at the example paths in the test configurations, or edit only the `weight` field to point to the corresponding local checkpoint. For example:

```bash
python test.py --config config/FSSD-12/1shot/fold0_test.yaml
python test.py --config config/FSSD-12/5shot/fold0_test.yaml
```

Evaluation refuses to run with an empty or missing checkpoint. Metrics include mean IoU, FB-IoU, and per-class IoU using the existing paper evaluation code. If `is_save: True`, prediction images are written below the configured `results/FSSD-12/` directory; when it is `False`, no result directory is created.

## Citation

If this code supports your work, please cite the FH-Net paper. Replace the placeholder below with the final bibliographic record after publication; no DOI is asserted here.

```bibtex
@article{fhnet,
  title   = {FH-Net: Frequency-Domain and Hierarchical Dual Enhancement for Few-Shot Surface Defect Segmentation},
  author  = {Fan, Lili and Yang, Zhanglin and Guo, Fan and Fang, Ping},
  journal = {PLOS ONE},
  year    = {forthcoming}
}
```

## License and third-party code

See `THIRD_PARTY_NOTICES.md` for the information that could be verified from the supplied files. A repository-wide open-source license has not yet been selected because upstream code provenance and license terms were not recorded. `LICENSE_PENDING.md` lists the required author checks; it must be resolved and replaced by a final `LICENSE` before public release.
