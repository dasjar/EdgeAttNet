# Required imports for plotting
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Polygon


def draw_cinched_box(ax, data, position, box_color='black', median_color='#FF7F24',
                     fill_color=None, width=0.3, pinch_frac=0.3, cinch_height_frac=0.08):
    """
    Function to draw a cinched waist style boxplot.
    """
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    iqr_min = q1 - 1.5 * iqr
    iqr_max = q3 + 1.5 * iqr
    filtered_data = [x for x in data if iqr_min <= x <= iqr_max]

    lower_whisker = np.min(filtered_data) if filtered_data else np.min(data)
    upper_whisker = np.max(filtered_data) if filtered_data else np.max(data)

    left = position - width / 2
    right = position + width / 2
    pinch_indent = width * pinch_frac
    cinch_height = iqr * cinch_height_frac
    top_cinch = median + cinch_height
    bottom_cinch = median - cinch_height

    verts = [
        (left, q1), (left, bottom_cinch), (left + pinch_indent, median),
        (left, top_cinch), (left, q3), (right, q3), (right, top_cinch),
        (right - pinch_indent, median), (right, bottom_cinch), (right, q1),
    ]
    polygon = Polygon(verts, closed=True, edgecolor=box_color,
                      facecolor=fill_color if fill_color else 'none', linewidth=1.5)
    ax.add_patch(polygon)

    ax.vlines(position, lower_whisker, q1, color='black', linewidth=1.2)
    ax.vlines(position, q3, upper_whisker, color='black', linewidth=1.2)

    cap_width = width * 0.25
    ax.hlines(lower_whisker, position - cap_width / 2, position + cap_width / 2, color='black', linewidth=1.2)
    ax.hlines(upper_whisker, position - cap_width / 2, position + cap_width / 2, color='black', linewidth=1.2)

    ax.hlines(median, left + pinch_indent, right - pinch_indent, color=median_color, linewidth=2)

    outliers = [x for x in data if x < iqr_min or x > iqr_max]
    if outliers:
        ax.plot([position] * len(outliers), outliers, 'o', markerfacecolor='none',
                markeredgecolor='black', markersize=6)

def generate_all_boxplots_in_row(model, dataloader, device,
                                 target_filenames,  # Added parameter for target filenames
                                 max_images=30,
                                 save_name="boxplots_edgeattnet.pdf"):
    """
    Generates a series of boxplots (modified with cinched waist) for IoU values 
    and saves the plot as a PDF file.
    """
    model.eval()
    spacing = 0.8
    group_width = 0.25
    inner_gap = 0.1  # updated from 0.08 to 0.1
    post_separator_gap = 0.15
    fig_width = max_images * (spacing + post_separator_gap)
    fig, ax = plt.subplots(figsize=(fig_width, 4.0), dpi=600)  # height 3.0 instead of 5.0

    image_counter = 0
    scales = [32, 16, 8, 4, 2, 1]
    separator_gap = 0.2
    separator_lines = []

    filenames_for_xaxis = []  # List to hold the filenames to display on the x-axis

    with torch.no_grad():
        for images, masks, filenames in tqdm(dataloader, desc="Generating Cinched Waist Boxplots"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs[0]).cpu().numpy()  # keep this as-is per your request
            preds = (probs > 0.5).astype(float)

            for i in range(images.size(0)):
                filename = filenames[i]

                # Only include filenames that are in target_filenames
                if filename not in target_filenames:
                    continue

                if image_counter >= max_images:
                    break

                gt = masks[i, 0].cpu().numpy() > 0.5
                pred = preds[i, 0]
                pairwise_ious = compute_pairwise_iou(gt, pred)

                if isinstance(pairwise_ious, (int, float)):
                    pairwise_ious = [pairwise_ious]
                if len(pairwise_ious) == 0:
                    print(f"Skipping {filename} due to insufficient pairwise IoU values")
                    pairwise_ious = []

                gt_labeled, pred_labeled, matches = match_objects(gt, pred, iou_threshold=0.5)
                multiscale_ious = []
                for gt_obj, pred_obj in matches:
                    gt_obj_mask = (gt_labeled == gt_obj)
                    pred_obj_mask = (pred_labeled == pred_obj)
                    gt_contour = extract_contour(gt_obj_mask)
                    pred_contour = extract_contour(pred_obj_mask)
                    miou = compute_miou_per_object(gt_contour, pred_contour, scales)
                    multiscale_ious.append(miou)

                if len(multiscale_ious) == 0:
                    multiscale_ious = []

                group_center = (image_counter + 1) * spacing + image_counter * post_separator_gap

                pairwise_pos = group_center - group_width / 2 - inner_gap / 2
                ms_pos = group_center + group_width / 2 + inner_gap / 2

                # Draw pairwise IoU boxplot
                if pairwise_ious:
                    draw_cinched_box(ax, pairwise_ious, position=pairwise_pos,
                                     box_color='#8A2BE2', median_color='#FF7F24',
                                     fill_color=None, width=0.3, pinch_frac=0.3)
                    ax.plot(pairwise_pos, np.mean(pairwise_ious), marker='^', color='#5ab4ac', markersize=8, linestyle='None')
                else:
                    ax.text(pairwise_pos, 0.5, 'Insufficient IoU', ha='center', va='center', color='red', fontsize=8)

                # Draw multiscale IoU boxplot
                if multiscale_ious:
                    draw_cinched_box(ax, multiscale_ious, position=ms_pos,
                                     box_color='black', median_color='#FF7F24',
                                     fill_color='#CD5B45', width=0.3, pinch_frac=0.3)
                    ax.plot(ms_pos, np.mean(multiscale_ious), marker='^', color='#5ab4ac', markersize=8, linestyle='None')
                else:
                    ax.text(ms_pos, 0.5, 'Insufficient IoU', ha='center', va='center', color='red', fontsize=8)

                separator_lines.append(ms_pos + group_width / 2 + separator_gap)
                filenames_for_xaxis.append(filename)

                image_counter += 1

            if image_counter >= max_images:
                break

    # Add separator lines to the plot
    if separator_lines:
        separator_lines = separator_lines[:-1]

    for x in separator_lines:
        ax.axvline(x, color='gray', linestyle='--', linewidth=2.0, alpha=0.7)

    # Customize plot axis labels and ticks
    ax.set_xlim(spacing * 0.5, (image_counter + 1) * spacing + (image_counter - 1) * post_separator_gap)
    ax.set_ylim(0.35, 1.05)
    ax.set_ylabel('IoU', fontsize=15)
    ax.set_xlabel('H-alpha Observation Timestamp', fontsize=14)

    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.6)

    # Set xticks to be at group centers
    xtick_positions = []
    for idx in range(image_counter):
        group_center = (idx + 1) * spacing + idx * post_separator_gap
        xtick_positions.append(group_center)

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(filenames_for_xaxis, rotation=83, ha='right', fontsize=8)
    ax.tick_params(axis='y', direction='out', length=6, width=1.5, labelsize=10)

    # Save and show the plot
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(save_name, bbox_inches='tight', format='pdf')
    plt.show()


# Example on how to use:
# Define the target filenames the user wants to include
target_filenames = [
    "010101-20220523234912Lh", "010203-20220710085152Th",
    "010401-20210726041650Uh", "010401-20220621185332Bh",
    "010401-20220721185332Bh", "020401-20220122085210Th",
    "040201-20210620163530Mh", "040201-20220124155050Bh",
    "040401-20220618185332Bh", "040401-20220714185352Th",
    "050101-20220622205632Bh", "050201-20210730044930Lh"
]

# Assuming 'model', 'dataloader', and 'device' are defined elsewhere in your code
generate_all_boxplots_in_row(
    model=model,
    dataloader=test_loader,
    device=device,
    target_filenames=target_filenames,  # Pass the list as an argument
    max_images=12,
    save_name="boxplots_edgeattnet.pdf"
)
