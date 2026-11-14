# WILLIE-MINI in PyHealth

Re-implementation of the **MINI model** from *WILLIE* (Wound Imaging, Localization, and Integrated Evaluation, arXiv 2610.05341) using PyHealth's `Dataset` / `Task` / `Model` abstractions. The model keeps the original module names, so the authors' public weights load directly and give identical outputs.

## Results

Public fold-0 checkpoint (`QianGroup/willie-weights`, `mini/willie_mini_fold0_best.pt`), evaluated through the PyHealth pipeline:

| Check | Reference | PyHealth |
|---|---|---|
| Max logit difference vs. original code (234 test images) | 0 | **0.0** |
| Prediction agreement with original code | 100% | **100%** |
| Classification accuracy, test (n=234) | 87.61% | **87.61%** |
| Classification accuracy, fold 0 val (n=216) | 84.72% | **84.72%** |
| Segmentation Dice, fold 0 val (n=202) | 79.18 | 79.47 (approximate) |
| Mask-derived box AP@0.5, fold 0 val | 81.66 | 81.29 (approximate) |

The unit tests (19) use synthetic images and a tiny stand-in backbone. Segmentation numbers are an approximate comparison (see the notebook, section 5). Only a 2-epoch classification smoke test was run for training; the full training run and the paper's 5-fold ensemble numbers are **not** reproduced.

## Robustness stress test (RQ3, partial)

The public fold-0 checkpoint evaluated on `cls_test` (n=234, AZH only) after six synthetic corruptions at three severities (notebook section 6). Clean accuracy: 87.61% (macro-F1 85.54%). Values are accuracy in %.

| Corruption | Mild | Moderate | Severe |
|---|---|---|---|
| darken (x0.7 / x0.5 / x0.3) | 90.17 | 88.46 | 83.33 |
| brighten (x1.3 / x1.6 / x2.0) | 88.03 | 85.90 | 82.48 |
| low contrast (x0.7 / x0.5 / x0.3) | 87.61 | 83.76 | 76.07 |
| Gaussian noise (sigma 8 / 16 / 32) | 87.18 | 86.75 | 86.75 |
| Gaussian blur (sigma 1 / 2 / 4 px) | 87.18 | 87.18 | 80.34 |
| JPEG compression (quality 50 / 20 / 5) | 88.03 | 87.18 | 62.39 |

- Tolerated: noise up to sigma 32, blur up to sigma 2, JPEG down to quality 20, mild brightness changes (differences within the bootstrap noise).
- Largest losses come from removing information: JPEG quality 5 (-25.2 points, 95% CI [19.2, 31.6]), low contrast x0.3 (-11.5, [6.8, 16.2]), blur sigma 4 (-7.3, [3.0, 11.5]).
- Macro-F1 falls faster than accuracy at the strongest blur, low-contrast, JPEG and brightening settings, which suggests the minority classes suffer more (per-class results not computed).
- Caveats: AZH-only test set (not a cross-source test), synthetic corruptions applied at model resolution, single checkpoint and fold, no augmentation-enhanced model to compare against, n=234 and uncorrected intervals over 18 conditions. JPEG quality 5 is an extreme setting.

## What is included

| File (in PyHealth) | Purpose |
|---|---|
| `pyhealth/models/willie_mini.py` | `WILLIEMini` (DINOv2 ViT-S/14, FPN, WA-CSA, MoE classifier, segmentation and detection heads) |
| `pyhealth/datasets/willie.py`, `datasets/configs/willie.yaml` | `WILLIEDataset` for AZH, Medetec and FUSeg with the public 5-fold indices |
| `pyhealth/tasks/willie_tasks.py` | `WILLIEClassification` (5 classes), `WILLIESegmentation` |
| `pyhealth/processors/wound_processors.py` | `WoundImageProcessor`, `WoundMaskProcessor` |
| `tests/core/test_willie.py` | 19 unit tests |
| `WILLIE_MINI_PyHealth_EN.ipynb` | End-to-end Colab notebook: download, install, test, verify |

## Quick start (Google Colab, T4)

1. Open `WILLIE_MINI_PyHealth_EN.ipynb` and run section 1: it downloads the data from HuggingFace (`QianGroup/willie-benchmark`) and the code from GitHub (`Qian-Group-HRI/Willie`) and verifies 3,535 files.
2. Section 3 clones PyHealth, writes the files above and appends one registration line to each `__init__.py`. Restart the runtime afterwards.
3. Sections 4 and 5 build the dataset and reproduce the table above.

```python
from pyhealth.datasets import WILLIEDataset
from pyhealth.models import WILLIEMini
from pyhealth.tasks import WILLIEClassification

dataset = WILLIEDataset(root="/content/Willie", folds_path=".../manifests/5fold_splits_v2.pt")
test_ds = dataset.set_task(WILLIEClassification(split="test"))
model = WILLIEMini(test_ds, pretrained_backbone=False)
model.load_original_checkpoint("willie_mini_fold0_best.pt")
```

## Progress against the project proposal

Proposal: *Reproducing and Stress-Testing WILLIE-MINI under Dataset and Acquisition Shifts* (CS 598 DL4H, Fall 2026).

| Proposal item | Status | Evidence / note |
|---|---|---|
| Data access and 3,535-file census (`verify_data.py`) | Done | 3,535 found, 0 missing |
| `WILLIEDataset` (manifest-based `BaseDataset`, yaml config) | Done | Section 4 of the notebook; metadata CSV plus public 5-fold indices |
| Classification and segmentation task classes | Done | `willie_tasks.py` (one file; the proposal named two) |
| `WILLIEMini` (`BaseModel`) with mask-to-box post-processing | Done | Weights load strictly; outputs identical to the original code (max logit diff 0.0) |
| Missing-label handling | Done | Loss is computed only for the targets present in a sample |
| Unit tests on synthetic data | Done | 19 tests |
| Example script and API docs | Partial | The notebook serves as the example; docstrings only |
| Audit of the released implementation | Done | See the inconsistency list below |
| Reproduce MINI results | Partial | Public fold-0 checkpoint reproduced exactly through PyHealth (test 87.61%, fold-0 val 84.72%, Dice and AP approximate). Training from scratch not reproduced; macro-F1 computed only on the original code path; IoU and macro-AUC not computed |
| Multi-task training recipe (weighted CE + BCE + Dice, EMA, augmentation, task sampling) | Not done | Only a classification-only 2-epoch smoke test |
| RQ1/RQ2: DINOv2+Linear and YOLOv8m/RT-DETR baselines, H1/H2 trends | Not done | |
| RQ3: robustness to photometric/compression shifts | Done (public checkpoint) | Notebook section 6: 6 corruptions x 3 severities on `cls_test`; accuracy, macro-F1, paired-bootstrap drop. See the table above |
| RQ3: augmentation-enhanced variant, source-aware sampling, cross-source test | Not done | Needs training; planned extension |

In short: the PyHealth integration (dataset, tasks, model, tests) and the faithful reproduction of the public MINI checkpoint are complete, and a corruption stress test (RQ3, partial) has been run on the public checkpoint; the baseline comparisons (RQ1/RQ2) and the training-based part of RQ3 remain.

## Notes and limitations

- The checkpoint's saved config (518x518 input, 2 experts) differs from the README of the original release (224x224, 8 experts); the checkpoint was treated as the source of truth.
- The original "AP@0.5" is precision x recall of mask-derived connected-component boxes, not COCO AP.
- No augmentation, class-balanced sampler or joint multi-task loss in the PyHealth training loop.
- The proposal states a 1,081-image classification pool, but the manifests and the fold file contain 1,080 (918 + 162); the difference is unexplained.
- The proposal drops mask components under 20 px; the original code uses a minimum area of 50, which this implementation follows.
- Detection manifests (810 + 400 = 1,210 rows) do not match the public fold file (976 + 244 = 1,220); detection was not evaluated.
- Checkpoints are loaded with `weights_only=False` (official file with numpy scalars). Do not do this for untrusted files.

## References

- WILLIE paper: arXiv 2610.05341. Code: https://github.com/Qian-Group-HRI/Willie
- Data: https://huggingface.co/datasets/QianGroup/willie-benchmark. Weights: https://huggingface.co/QianGroup/willie-weights
- PyHealth: https://github.com/sunlabuiuc/PyHealth
