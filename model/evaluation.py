import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
from utils import compute_pairwise_iou, compute_dataset_multiscale_iou, collect_predictions_and_labels, compute_average_recall_iou_thresholds

# Average Precision 
def compute_ap(y_true, y_score):
    """
    Computes the average precision score and precision-recall curve.
    Args:
        y_true (np.ndarray): Ground truth values.
        y_score (np.ndarray): Predicted scores.
    Returns:
        tuple: Average precision score, precision, recall, and thresholds.
    """
    y_true_flat = y_true.flatten()
    y_score_flat = y_score.flatten()
    precision, recall, thresholds = precision_recall_curve(y_true_flat, y_score_flat)
    ap = average_precision_score(y_true_flat, y_score_flat)
    return ap, precision, recall, thresholds

# Average Recall over thresholds
def compute_average_recall_iou_thresholds(y_true, y_pred, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Computes the average recall over multiple IoU thresholds.
    Args:
        y_true (np.ndarray): Ground truth masks.
        y_pred (np.ndarray): Predicted masks.
        thresholds (np.ndarray): List of IoU thresholds.
    Returns:
        float: Average recall score.
    """
    recalls = []
    for thresh in thresholds:
        recall_sum = 0.0
        for i in range(len(y_true)):
            iou = compute_pairwise_iou(y_true[i, 0] > 0.5, y_pred[i, 0] > 0.5, iou_threshold=thresh)
            recall_sum += len(iou)
        recalls.append(recall_sum / len(y_true))
    return np.mean(recalls)

# Main evaluation function
def evaluate_model(model, test_loader, device):
    """
    Main function to evaluate the model on the test set.
    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): Device for model inference.
    Returns:
        dict: Evaluation results (AP, Recall, mIoU scores).
    """
    start_time = time.time()
    print("Collecting predictions")    
    y_true, y_score, preds_05, y_targets = collect_predictions_and_labels(model, test_loader, device)
    
    print("Computing Average Precision")  
    ap, precision, recall, thresholds_raw = compute_ap(y_true, y_score)
    
    thresholds = np.arange(0.5, 1.0, 0.05)
    print("Computing Average Recall (object-level)")
    ar = compute_average_recall_iou_thresholds(y_targets, preds_05, thresholds)
    
    print("Computing Pairwise Mean IoU (old method)")   
    pairwise_miou = compute_dataset_pairwise_miou(y_targets, preds_05, iou_threshold=0.5)

    print("Computing Multiscale IoU (NEW paper-aligned method)")   
    ms_iou = compute_dataset_multiscale_iou(y_targets, preds_05, iou_threshold=0.5)

    duration = time.time() - start_time
    print(f"Evaluation done in {duration:.2f}s")
    
    return {
        "AP": ap,
        "Average Recall": ar,
        "Pairwise mIoU (old)": pairwise_miou,
        "Multiscale IoU (new)": ms_iou,
    }
