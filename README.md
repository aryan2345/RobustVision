# RobustVision: Image Classification Under Visual Corruption

A controlled computer vision experiment studying whether corruption-aware augmentation improves CNN robustness to Gaussian noise and partial occlusion on CIFAR-10.

## Research question
Does training with simulated visual corruption improve robustness relative to standard augmentation, and does it trade off clean-image accuracy?

## Setup
Python 3.10+ recommended. In a terminal inside this repository:

```bash
python -m venv .venv
```
Activate on Windows PowerShell: `.venv\Scripts\Activate.ps1` (on macOS/Linux: `source .venv/bin/activate`).

```bash
python -m pip install -r requirements.txt
python train.py --smoke --output smoke_results
python train.py --epochs 15 --output results
```
The full run downloads CIFAR-10 on first use; internet access is required. CUDA is automatically selected when available. CPU training can be substantially slower. The smoke test uses synthetic images and verifies execution only, not model quality.

## Experimental design
- Fixed seeded split: 45,000 training, 5,000 validation, 10,000 held-out test images.
- Identical six-convolution CNNs with batch normalization, ReLU, pooling, global average pooling and 10 logits.
- Both models use random crops and horizontal flips. Robust model additionally receives noise or square occlusion on 50% of training samples.
- Gaussian noise standard deviations: 0.05–0.30 on pixel range [0,1]. Square occlusion side fractions: 0.125–0.50 (area fraction is the square of side fraction). Occlusion uses neutral gray fill.
- Same architecture, initialization seed, optimizer, training budget and data split. Best checkpoint chosen on clean validation accuracy, not test accuracy.
- Identical deterministic test corruptions for both models at each severity.
- AdamW, cross-entropy loss, batch size 128 by default.

## Outputs
`results/metrics.csv`: test accuracy, macro F1 and loss for each model/condition.
`results/training.csv`: training loss and validation accuracy by epoch.
`results/*_robustness.png`: accuracy versus corruption severity.
`results/*_confusion.png` and CSV: confusion matrices for clean and strongest occlusion conditions.
`results/*.pt`: model weights. `results/run.json`: experiment settings and data sizes.
Class IDs: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (0–9).

## Analysis to complete after training
Compare clean accuracy and absolute percentage-point accuracy degradation at each severity. Examine per-class confusion patterns. Determine whether augmentation improves corrupted accuracy while sacrificing clean accuracy. Repeat with `--seed 43` and `--seed 44` (separate output folders) and report mean and standard deviation. Do not tune on the test set. Results are not yet established; the included smoke test confirms only the execution pipeline. Synthetic corruption does not represent all real-world distribution shifts, and these experiments do not demonstrate adversarial robustness.

## Reference
Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
