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

## Notes and limitations

- The checkpoint's saved config (518x518 input, 2 experts) differs from the README of the original release (224x224, 8 experts); the checkpoint was treated as the source of truth.
- The original "AP@0.5" is precision x recall of mask-derived connected-component boxes, not COCO AP.
- No augmentation, class-balanced sampler or joint multi-task loss in the PyHealth training loop.
- Detection manifests (810 + 400 = 1,210 rows) do not match the public fold file (976 + 244 = 1,220); detection was not evaluated.
- Checkpoints are loaded with `weights_only=False` (official file with numpy scalars). Do not do this for untrusted files.

## References

- WILLIE paper: arXiv 2610.05341. Code: https://github.com/Qian-Group-HRI/Willie
- Data: https://huggingface.co/datasets/QianGroup/willie-benchmark. Weights: https://huggingface.co/QianGroup/willie-weights
- PyHealth: https://github.com/sunlabuiuc/PyHealth

