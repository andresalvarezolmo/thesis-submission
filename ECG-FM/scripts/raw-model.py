
from fairseq_signals.utils.store import MemmapReader
from imblearn.metrics import specificity_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, confusion_matrix
import os
import pandas as pd
import torch

root = '/home/datasets/code_15/subset'
experiment_root = '/home/datasets/code_15/experiments/subset'
fairseq_signals_root = '/home/playground/fairseq-signals'
fairseq_signals_root = fairseq_signals_root.rstrip('/')
fairseq_signals_root

segmented_path = f'/home/datasets/code_15/subset/test_segmented_split.csv'
segmented_split = pd.read_csv(segmented_path,
    index_col='idx',
)

run = "1"
segmented_path = f'/home/datasets/code_15/subset/test_segmented_split.csv'
segmented_split = pd.read_csv(segmented_path,
    index_col='idx',
)

each_experiment_path = os.path.join(experiment_root, "raw", run)
os.makedirs(each_experiment_path, exist_ok=True)

model_path = f"/home/playground/ECG-FM/experiments/raw/5/checkpoint1.pt"
print(each_experiment_path)


inference_cmd = f"""fairseq-hydra-inference \\
    task.data="/home/datasets/code_15/subset/manifests" \\
    common_eval.path="{model_path}" \\
    common_eval.results_path="{each_experiment_path}" \\
    model.num_labels=6 \\
    dataset.valid_subset="test" \\
    dataset.batch_size=10 \\
    dataset.num_workers=3 \\
    dataset.disable_validation=false \\
    distributed_training.distributed_world_size=1 \\
    distributed_training.find_unused_parameters=True \\
    --config-dir "/home/playground/ECG-FM/ckpts/" \\
    --config-name physionet_finetuned
"""

os.system(inference_cmd)

code15_label_def = pd.read_csv(
    os.path.join('/home/datasets/code_15/subset/label_def.csv'),
     index_col='name',
)
code15_label_names = code15_label_def.index

logits = MemmapReader.from_header(f"{each_experiment_path}/outputs_test.npy")[:]

pred = pd.DataFrame(
    torch.sigmoid(torch.tensor(logits)).numpy(),
    columns=code15_label_names,
)

pred = segmented_split.reset_index().join(pred, how='left').set_index('idx')

pred_thresh = pred.copy()
pred_thresh[code15_label_names] = pred_thresh[code15_label_names] > 0.5

pred_thresh['labels'] = pred_thresh[code15_label_names].apply(
    lambda row: ', '.join(row.index[row]),
    axis=1,
)
pred_thresh['labels']


code15_label_def = pd.read_csv(
    os.path.join('/home/datasets/code_15/subset/label_def.csv'),
     index_col='name',
)
code15_label_names = code15_label_def.index
code_15_label_def = pd.read_csv("/home/playground/ECG-FM/data/code_15/labels/label_def.csv",
     index_col='name',
)
code_15_label_names = code_15_label_def.index
code_15_label_def

label_mapping = {
    'RBBB': 'RBBB',
    'LBBB': 'LBBB',
    'SB': 'SB',
    'ST': 'ST',
    'AF': 'AF',
    'normal_ecg': 'normal_ecg'
}

code15_label_def['name_mapped'] = code15_label_def.index.map(label_mapping)
code15_label_def

pred_mapped = pred.copy()
pred_mapped.drop(set(code15_label_names) - set(label_mapping.keys()), axis=1, inplace=True)
pred_mapped.rename(label_mapping, axis=1, inplace=True)

pred_thresh_mapped = pred_thresh.copy()
pred_thresh_mapped.drop(set(code15_label_names) - set(label_mapping.keys()), axis=1, inplace=True)
pred_thresh_mapped.rename(label_mapping, axis=1, inplace=True)
pred_thresh_mapped['predicted'] = pred_thresh_mapped[label_mapping.values()].apply(
    lambda row: ', '.join(row.index[row]),
    axis=1,
)
true_labels = pd.read_csv(os.path.join('/home/datasets/code_15/subset/ground_truth_test_labels.csv'), index_col='idx')
true_labels['actual'] = true_labels[label_mapping.values()].apply(
    lambda row: ', '.join(row.index[row]),
    axis=1,
)

pred_thresh_mapped[['predicted']].join(true_labels[['actual']], how='left')

comparison = pred_thresh_mapped[['predicted']].join(true_labels[['actual']], how='left')

accuracy = (comparison['predicted'] == comparison['actual']).mean()
print(f"Overall accuracy: {accuracy:.2%}")

y_true = comparison['actual']
y_pred = comparison['predicted']

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall    = recall_score(y_true, y_pred, average='weighted')
f1        = f1_score(y_true, y_pred, average='weighted')
specificity = specificity_score(y_true, y_pred, average='weighted')

print(f"Accuracy:    {accuracy:.3f}")
print(f"Precision:   {precision:.3f}")
print(f"Recall:      {recall:.3f}")
print(f"Specificity: {specificity:.3f}")
print(f"F1 score:    {f1:.3f}")

y_true_str = comparison['actual']
y_pred_str = comparison['predicted']

y_true_list = [labels.split(", ") for labels in y_true_str]
y_pred_list = [labels.split(", ") for labels in y_pred_str]

mlb = MultiLabelBinarizer()
y_true_bin = mlb.fit_transform(y_true_list)
y_pred_bin = mlb.transform(y_pred_list)

class_names = mlb.classes_

results = []

for i, cls_name in enumerate(class_names):
    y_true_col = y_true_bin[:, i]
    y_pred_col = y_pred_bin[:, i]
    
    tn, fp, fn, tp = confusion_matrix(y_true_col, y_pred_col).ravel()
    total = tp + tn + fp + fn
    prevalence = (tp + fn) / total
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    accuracy = (tp + tn) / total

    results.append({
        'class': cls_name,
        'prevalence': round(prevalence, 3),
        'f1': round(f1, 3),
        'precision': round(precision, 3),
        'recall': round(sensitivity, 3),
        'specificity': round(specificity, 3),
        'accuracy': round(accuracy, 3)
    })

metrics_df = pd.DataFrame(results)
print(metrics_df)
with open(f"/home/playground/ECG-FM/results/raw/raw-results-{run}.txt", "w") as f:
    f.write(metrics_df.drop(columns=['prevalence']).to_string(index=False))


