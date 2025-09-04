import numpy as np
import cv2
from skimage.measure import label
from tqdm import tqdm


# Extract contour from binary mask 
def extract_contour(mask_np):
    """
    Extracts contour from the binary mask using Canny edge detector.
    Args:
        mask_np (np.ndarray): Binary mask array.
    Returns:
        np.ndarray: Contour of the binary mask.
    """
    mask_uint8 = (mask_np.astype(np.uint8)) * 255
    edges = cv2.Canny(mask_uint8, threshold1=50, threshold2=150)
    contour = edges > 0
    return contour

Downsample mask using box count 
def box_count_downsample(mask_np, cell_size):
    """
    Downsamples the mask using box counting method.
    Args:
        mask_np (np.ndarray): The binary mask array.
        cell_size (int): The size of the box for downsampling.
    Returns:
        np.ndarray: Downsampled mask.
    """
    H, W = mask_np.shape
    new_H = H // cell_size
    new_W = W // cell_size
    if new_H == 0 or new_W == 0:
        return mask_np
    mask_cropped = mask_np[:new_H * cell_size, :new_W * cell_size]
    blocks = mask_cropped.reshape(new_H, cell_size, new_W, cell_size)
    blocks = blocks.max(axis=(1, 3))
    return blocks

#Intersection ratio for two masks
def compute_intersection_ratio(gt_cells, pred_cells):
    """
    Computes the intersection ratio between ground truth and predicted mask.
    Args:
        gt_cells (np.ndarray): Ground truth mask.
        pred_cells (np.ndarray): Predicted mask.
    Returns:
        float: Intersection ratio.
    """
    intersection = np.logical_and(gt_cells, pred_cells).sum()
    gt_area = gt_cells.sum()
    if gt_area == 0:
        return 1.0
    return intersection / gt_area

#Match objects (instance-level) between GT and prediction 
def match_objects(gt_mask, pred_mask, iou_threshold=0.5):
    """
    Matches objects (instances) between the ground truth and predicted masks.
    Args:
        gt_mask (np.ndarray): Ground truth mask.
        pred_mask (np.ndarray): Predicted mask.
        iou_threshold (float): IoU threshold for matching.
    Returns:
        tuple: Labeled ground truth and prediction, and matching results.
    """
    gt_labeled, n_gt = connected_components(gt_mask)
    pred_labeled, n_pred = connected_components(pred_mask)
    matches = []
    matched_pred = set()
    for gt_obj in range(1, n_gt + 1):
        gt_component = (gt_labeled == gt_obj)
        best_iou = 0
        best_pred_obj = None
        for pred_obj in range(1, n_pred + 1):
            if pred_obj in matched_pred:
                continue
            pred_component = (pred_labeled == pred_obj)
            intersection = np.logical_and(gt_component, pred_component).sum()
            union = np.logical_or(gt_component, pred_component).sum()
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_pred_obj = pred_obj
        if best_iou >= iou_threshold and best_pred_obj is not None:
            matches.append((gt_obj, best_pred_obj))
            matched_pred.add(best_pred_obj)
    return gt_labeled, pred_labeled, matches
