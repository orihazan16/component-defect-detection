# Component Defect Detection

Junior Data Analyst task: exploring photos of manufactured components, extracting simple image features, and building a first solution that flags suspicious (possibly defective) images.

## Files

| File | Contents |
| --- | --- |
| `component_defect_detection.ipynb` | Parts A-D in one notebook: data exploration, six sample images and two image features (`local_anomaly_score`, `contour_complexity`), the golden-template model with a dev/validation split, confusion matrix, precision/recall, one analysed error, and `predictions.csv` |
| `features.py` | Golden-template pipeline used by Part C: alignment (ORB + RANSAC), template, difference map, features |
| `part_e.txt` | Part E - conclusions and use of AI (Hebrew) |
| `predictions.csv` | 72 test predictions: `image_id,predicted_defect,score` |
| `train.csv`, `test.csv`, `images/` | Input data |

## How to run

Python 3.10+.

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `component_defect_detection.ipynb` from the repository root and run all cells top to bottom. It rewrites `predictions.csv`. The seeds are fixed, so the output is the same on every run.

Note: scikit-learn 1.5 does not work with numpy 2.x (`numpy.dtype size changed` on import), which is why `requirements.txt` asks for scikit-learn 1.6 or later.

## Results (validation split only)

| | |
| --- | --- |
| Precision | 1.00 |
| Recall | 0.82 |
| ROC AUC | 0.92 |

The validation split has 51 images, so these numbers move between random splits. Averaged over 10 splits the same pipeline gives precision 0.94, recall 0.74 and AUC 0.87.

How data leakage is avoided: the split comes first, and the reference image, golden template, scaler and model are all built from the development split only. `test.csv` is used only to produce predictions.
