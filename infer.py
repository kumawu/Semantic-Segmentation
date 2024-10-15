import torch
import random
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_fill_holes, label
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from skimage.morphology import erosion, disk
from skimage import filters, measure
import cv2

from sam2.sam2_image_predictor import SAM2ImagePredictor

# Timing decorator
def timeit(func):
    """Decorator for measuring execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result
    return wrapper

@timeit
def apply_morphology(mask, kernel_size=5):
    """Applies morphological operations to clean up a binary mask."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_open = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    mask_cleaned = binary_fill_holes(mask_closed).astype(np.uint8)
    
    return mask_cleaned

@timeit
def select_points_with_spread(heatmap, num_points, distance_threshold, heatmap_threshold):
    """Selects spread-out points on a heatmap above a threshold."""
    high_value_indices = np.argwhere(heatmap > heatmap_threshold)
    selected_points = [high_value_indices[0]]
    for point in high_value_indices[1:]:
        if len(selected_points) >= num_points:
            break
        selected_array = np.array(selected_points)
        distances = np.sqrt(np.sum((selected_array - point) ** 2, axis=1))
        if np.all(distances >= distance_threshold):
            selected_points.append(point)
    
    return np.array(selected_points)

@timeit
def select_points_with_spread_with_shuffle(heatmap, num_points, distance_threshold, heatmap_threshold):
    """Randomly shuffles and selects spread-out points from a heatmap."""
    high_value_indices = np.argwhere(heatmap > heatmap_threshold)
    random.shuffle(high_value_indices)
    
    selected_points = [high_value_indices[0]]
    selected_array = np.array(selected_points)
    
    distance_threshold_sq = distance_threshold ** 2

    for point in high_value_indices[1:]:
        if len(selected_points) >= num_points:
            break
        
        distances_sq = np.sum((selected_array - point) ** 2, axis=1)
        
        if np.all(distances_sq >= distance_threshold_sq):
            selected_points.append(point)
            selected_array = np.vstack([selected_array, point])
    
    return np.array(selected_points)

def select_low_gradient_high_value_points_limited(heatmap, gradient_threshold=30, value_threshold=0.95, erosion_radius=10, grid_size=(600, 600), max_points_per_grid=1):
    """
    Selects low-gradient, high-value points with limits per grid.
    Combines heatmap values and gradients to pick significant points.
    """
    smooth_heatmap = filters.gaussian(heatmap, sigma=2)
    threshold_strong = np.percentile(smooth_heatmap, value_threshold * 100)
    strong_binary_map = smooth_heatmap > threshold_strong

    eroded_map = erosion(strong_binary_map, disk(erosion_radius))

    gradient_y, gradient_x = np.gradient(smooth_heatmap)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    low_gradient_threshold = np.percentile(gradient_magnitude, gradient_threshold)
    low_gradient_map = gradient_magnitude < low_gradient_threshold

    combined_map = eroded_map & low_gradient_map
    labeled_combined_regions, num_features_combined = label(combined_map)

    combined_regions = measure.regionprops(labeled_combined_regions, intensity_image=smooth_heatmap)
    combined_centroids = [(int(region.centroid[0]), int(region.centroid[1])) for region in combined_regions]

    grid_height, grid_width = grid_size
    heatmap_height, heatmap_width = heatmap.shape

    grid_points = {}

    for centroid in combined_centroids:
        row, col = centroid
        grid_row = row // grid_height
        grid_col = col // grid_width

        grid_key = (grid_row, grid_col)

        if grid_key not in grid_points:
            grid_points[grid_key] = []

        grid_points[grid_key].append(centroid)

    final_selected_points = []
    for grid_key, points in grid_points.items():
        points_sorted_by_value = sorted(points, key=lambda p: smooth_heatmap[p[0], p[1]], reverse=True)
        final_selected_points.extend(points_sorted_by_value[:max_points_per_grid])
    
    return final_selected_points

@timeit
def process_image_with_clipseg(image, prompts, processor, model, num_points, distance_threshold, heatmap_threshold):
    """Processes an image using CLIPSeg model to generate segmentation heatmaps."""
    inputs = processor(text=prompts, images=[image] * len(prompts), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)

    all_coords = []
    results = []
    orig_size = image.size
    for i, prompt in enumerate(prompts):
        heatmap = torch.sigmoid(preds[i][0]).cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_resized = np.array(Image.fromarray(heatmap).resize(orig_size, Image.NEAREST))
        top_coords = select_low_gradient_high_value_points_limited(heatmap_resized, value_threshold=heatmap_threshold)
        all_coords.append(top_coords)
        results.append({
            'image': image,
            'heatmap_resized': heatmap_resized,
            'top_coords': top_coords,
            'prompt': prompt,
            'index': i,
            'orig_size': orig_size,
        })
    return all_coords, results

@timeit
def plot_heatmap_and_save(image, heatmap_resized, top_coords, prompt, index, orig_size, save_path):
    """Plots heatmap and saves the result as an image file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(heatmap_resized, cmap='hot', interpolation='nearest')
    ax[0].scatter([c[1] for c in top_coords], [c[0] for c in top_coords], color='blue')
    ax[0].set_title(f"Heatmap for '{prompt}'")
    ax[0].axis('off')
    ax[1].imshow(image.resize(orig_size))
    ax[1].scatter([c[1] for c in top_coords], [c[0] for c in top_coords], color='blue')
    ax[1].set_title(f"Original Image for '{prompt}'")
    ax[1].axis('off')
    plt.savefig(save_path)
    plt.close(fig)
    
@timeit
def apply_sam_and_overlay_masks(image_np, all_coords, prompts, sam_predictor, colors, kernel_size):
    """Applies SAM for segmentation and overlays masks for multiple prompts on the image."""
    overlay = np.zeros_like(image_np)
    h, w, _ = image_np.shape
    num_prompts = len(prompts)

    combined_mask = np.zeros((h, w, num_prompts), dtype=np.uint8)
    final_mask = np.zeros((h, w), dtype=np.uint8)

    for i, (prompt, prompt_coords) in enumerate(zip(prompts, all_coords)):
        input_prompts = {
            "point_coords": [(point[1], point[0]) for point in prompt_coords],
            "point_labels": [1] * len(prompt_coords)
        }
        
        # Apply SAM to generate masks for each prompt
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = sam_predictor.predict(**input_prompts)
            
        combined_mask[:, :, i] = np.bitwise_or.reduce(masks.astype(np.uint8), axis=0)
        combined_mask[:, :, i] = apply_morphology(combined_mask[:, :, i], kernel_size)
        
    for i in range(num_prompts):
        mask_cleaned = combined_mask[:, :, i]
        mask_cleaned_rgb = np.repeat(mask_cleaned[:, :, np.newaxis], 3, axis=2)
        colored_mask = mask_cleaned_rgb * np.array(colors[i])
        overlay[mask_cleaned > 0] = colored_mask[mask_cleaned > 0]


    for i in range(combined_mask.shape[2]):
        final_mask[combined_mask[:, :, i] == 1] = i+1
        
    final_overlay = (0.5 * image_np + 0.5 * overlay).astype(np.uint8)
    
    return final_overlay, final_mask

@timeit
def plot_and_save_final_image(image_np, final_image, save_dir):
    """Plots the final segmented image and saves it to disk."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image_np)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    ax[1].imshow(final_image)
    ax[1].set_title('Segmented Image with CLIPSeg + SAM')
    ax[1].axis('off')
    os.makedirs(save_dir, exist_ok=True)
    final_image_path = os.path.join(save_dir, 'segmented_image.png')
    Image.fromarray(final_image).save(final_image_path)
    
    
def load_default_colors(file_path='colors.txt'):
    """Loads the default colors from a text file and returns them as a semicolon-separated string."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            colors = [line.strip() for line in f if line.strip()]  # Read and strip each line
        return ';'.join(colors)
    
    
def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    sam_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")

    image = Image.open(args.image_path).convert("RGB")
    image_name = args.image_path.split(os.sep)[-1].split('.')[0]
    image_np = np.array(image)
    sam_predictor.set_image(image)
    
    prompts = args.prompts.split(',')
    colors = [tuple(map(int, color.split(','))) for color in args.colors.split(';')]

    all_coords, results = process_image_with_clipseg(
        image, prompts, processor, model, args.num_points, 
        args.distance_threshold, args.heatmap_threshold
    )

    # Now call plot_heatmap_and_save in the main method
    for result in results:
        prompt = result['prompt']
        save_path = os.path.join(args.save_dir, f"heatmap_{prompt}_{image_name}.png")
        plot_heatmap_and_save(
            result['image'], result['heatmap_resized'], result['top_coords'], 
            result['prompt'], result['index'], result['orig_size'], save_path
        )

    final_image, final_mask = apply_sam_and_overlay_masks(
        image_np, all_coords, prompts, sam_predictor, colors, args.kernel_size
    )
    
    heatmap_log = np.exp(result['heatmap_resized'])
    heatmap_normalized = (heatmap_log - np.min(heatmap_log)) / (np.max(heatmap_log) - np.min(heatmap_log))
    
    # plot_and_save_final_image(image_np, final_image, args.save_dir)
    Image.fromarray(final_image).save(\
        os.path.join(args.save_dir, args.image_path.split(os.sep)[-1].split('.')[0] + "_segmented.png")
    )
    Image.fromarray(final_mask.astype(np.uint8)).save(\
        os.path.join(args.save_dir, args.image_path.split(os.sep)[-1].split('.')[0] + "_mask.png")
    )
    Image.fromarray((heatmap_normalized * 255).astype(np.uint8)).save(\
        os.path.join(args.save_dir, args.image_path.split(os.sep)[-1].split('.')[0] + "_heatmap.png")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--prompts', type=str, required=True, help='Comma-separated list of prompts for segmentation')
    parser.add_argument('--num_points', type=int, default=20, help='Number of points to select for each prompt')
    parser.add_argument('--distance_threshold', type=int, default=1500, help='Minimum distance between selected points')
    parser.add_argument('--colors', type=str, default=load_default_colors(), help='Semicolon-separated RGB colors for each prompt in the format R,G,B')
    parser.add_argument('--save_dir', type=str, default='./saved_figures', help='Directory to save the output images')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for morphological operations to clean masks')
    parser.add_argument('--heatmap_threshold', type=float, default=0.6, help='Threshold for selecting high-value heatmap points')
    parser.add_argument('--seed', type=int, default=24, help='Seed for random operations to ensure reproducibility')
    args = parser.parse_args()
    main(args)
