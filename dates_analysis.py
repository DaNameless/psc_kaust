# module: date_analysis.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from pathlib import Path
import math
import argparse
import scipy.ndimage as ndi
from skimage import img_as_float
from skimage.filters import threshold_local
from scipy import ndimage as ndi

def parse_filename(filename):
    name = Path(filename).stem
    match = re.match(r"([A-Za-z]+)_(\d+)-(\d+)_([\w\s]+)__", name)
    if not match:
        raise ValueError(f"Nombre de archivo no reconocido: {filename}")
    variety = match.group(1)
    id_start = int(match.group(2))
    id_end = int(match.group(3))
    light = match.group(4).strip()
    return variety, id_start, id_end, light

from sklearn.cluster import DBSCAN
import numpy as np
import math

def sort_components_by_row(components, eps_fraction=0.1):
    """
    Sorts components row-wise using DBSCAN clustering on Y-centroids.
    `eps_fraction` determines sensitivity to row height.
    """
    if not components:
        return []

    centroids_y = np.array([c["centroid_y"] for c in components]).reshape(-1, 1)
    image_height = max(centroids_y)[0] if centroids_y.size > 0 else 1
    eps = image_height * eps_fraction  # fraction of image height
    
    clustering = DBSCAN(eps=eps, min_samples=1).fit(centroids_y)
    
    # Add row label to each component
    for comp, label in zip(components, clustering.labels_):
        comp["row"] = label

    # Sort by row (top to bottom), then by x (left to right)
    components.sort(key=lambda c: (c["row"], c["centroid_x"]))
    return components


def extract_features_from_mask(img_rgb, mask, assigned_ids):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        component_mask = (labels == i).astype(np.uint8)
        masked_pixels = img_rgb[component_mask == 1]
        R_vals = masked_pixels[:, 0]
        G_vals = masked_pixels[:, 1]
        B_vals = masked_pixels[:, 2]
        comp_data = {
            "label": i,
            "area": area,
            "centroid_x": cx,
            "centroid_y": cy,
            "mean_R": np.mean(R_vals),
            "mean_G": np.mean(G_vals),
            "mean_B": np.mean(B_vals),
            "median_R": np.median(R_vals),
            "median_G": np.median(G_vals),
            "median_B": np.median(B_vals),
            "std_R": np.std(R_vals),
            "std_G": np.std(G_vals),
            "std_B": np.std(B_vals),
            "mask": component_mask,
            "bbox": (x, y, w, h),
        }
        components.append(comp_data)
    #row_tolerance = img_rgb.shape[0] * 0.25
    #components.sort(key=lambda c: (math.floor(c["centroid_y"] / row_tolerance), c["centroid_x"])
    components = sort_components_by_row(components)
    labeled_data = []
    if len(components) != len(assigned_ids):
        print(f"WARNING: Found {len(components)} objects, expected {len(assigned_ids)}.")
    for i, comp in enumerate(components):
        labeled_data.append({
            "ID": assigned_ids[i] if i < len(assigned_ids) else f"unmatched_{i}",
            "area": comp["area"],
            "centroid_x": comp["centroid_x"],
            "centroid_y": comp["centroid_y"],
            "mean_R": comp["mean_R"],
            "mean_G": comp["mean_G"],
            "mean_B": comp["mean_B"],
            "median_R": comp["median_R"],
            "median_G": comp["median_G"],
            "median_B": comp["median_B"],
            "std_R": comp["std_R"],
            "std_G": comp["std_G"],
            "std_B": comp["std_B"],
            "bbox": comp["bbox"],
        })
    return labeled_data



def process_image(path, output_dir):
    variety, id_start, id_end, light = parse_filename(path)
    assigned_ids = list(range(id_start, id_end + 1))
    print(variety, id_start, id_end, light)

    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(img_rgb)
    B_norm = img_as_float(B)

    # Adaptive threshold
    block_size = 621
    adaptive_thresh = threshold_local(B_norm, block_size=block_size, offset=0.0445)
    binary_mask = (B_norm < adaptive_thresh).astype(np.uint8)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Distance transform + watershed
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.12* dist_transform.max(), 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(binary_mask, sure_fg)

    # Marker labeling
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0
    markers = cv2.watershed(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), markers)

    # Create filtered mask from separated regions
    filtered_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    for label in range(2, num_markers + 2):  # markers start from 2
        region_mask = (markers == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 70000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = area / hull_area
            extent = area / (w * h)
            if (
                circularity > 0.0 and
                0.6 < aspect_ratio < 3 and
                solidity > 0.5 and
                extent > 0.1
            ):
                cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Feature extraction and labeling
    labeled_data = extract_features_from_mask(img_rgb, filtered_mask, assigned_ids)

    # Draw bounding boxes and IDs
    for i, row in enumerate(labeled_data):
        x, y, w, h = row["bbox"]
        id_label = str(row["ID"])
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=5)
        font_scale = 5
        thickness = 10
        text_size = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = x
        text_y = y - 10 if y - 10 > 10 else y + 20
        cv2.putText(
            img_rgb,
            id_label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 0, 0),
            thickness,
            lineType=cv2.LINE_AA
        )

    global_stats_from_mask(img_rgb, filtered_mask)

    # Save labeled image
    filename_stem = Path(path).stem
    labeled_img_path = os.path.join(output_dir, "labeled", f"{filename_stem}_labeled.png")
    os.makedirs(os.path.dirname(labeled_img_path), exist_ok=True)
    plt.figure()
    plt.imshow(img_rgb)
    plt.title("Identified Dates with Labels")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(labeled_img_path)
    plt.close()

    # Add metadata
    for row in labeled_data:
        row.update({
            "variety": variety,
            "light": light,
            "filename": os.path.basename(path),
        })
    return labeled_data

def global_stats_from_mask(img_rgb, filtered_mask, show_hist=True, bins=256, output_dir=None, filename_stem=None):
    masked_pixels = img_rgb[filtered_mask == 255]
    R_vals = masked_pixels[:, 0]
    G_vals = masked_pixels[:, 1]
    B_vals = masked_pixels[:, 2]
    mean_R = np.mean(R_vals)
    mean_G = np.mean(G_vals)
    mean_B = np.mean(B_vals)
    print("Global Mean RGB (dates only):", (mean_R, mean_G, mean_B))
    if show_hist and output_dir and filename_stem:
        hist_path = os.path.join(output_dir, "histograms", f"{filename_stem}_hist.png")
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.hist(R_vals, bins=bins, color='red', alpha=0.6, label='R')
        plt.hist(G_vals, bins=bins, color='green', alpha=0.6, label='G')
        plt.hist(B_vals, bins=bins, color='blue', alpha=0.6, label='B')
        plt.title("RGB Histograms (Masked Region)")
        plt.xlabel("Pixel Value")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
    return (mean_R, mean_G, mean_B)


def process_directory(directory, output_dir):
    all_data = []
    for fname in os.listdir(directory):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(directory, fname)
            try:
                print(f"Processing {fname}...")
                data = process_image(full_path, output_dir)
                all_data.extend(data)
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    return pd.DataFrame(all_data)

def main():

    parser = argparse.ArgumentParser(description="Analyze date images in a folder.")
    parser.add_argument("--input_folder", help="Path to the folder containing the images.")
    parser.add_argument("--output_csv", default="results.csv", help="CSV file to save results.")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save labeled images and histograms.")
    args = parser.parse_args()


    print(f"Analyzing images in: {args.input_folder}")
    df = process_directory(args.input_folder, args.output_dir)

    print("\nAnalysis complete. Summary:")
    print(df.head())

    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, args.output_csv), index=False, float_format="%.6f")
    print(f"\nResults saved to {os.path.join(args.output_dir, args.output_csv)}")


if __name__ == "__main__":
    main()