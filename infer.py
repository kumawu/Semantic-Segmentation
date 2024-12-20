import torch
import random
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from scipy.ndimage import binary_fill_holes, label
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from skimage.morphology import erosion, disk
from skimage import filters, measure
import cv2

from sam2.sam2_image_predictor import SAM2ImagePredictor

# Timing decorator
def timeit(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result
    return wrapper

@timeit
def apply_morphology(mask, kernel_size=5):
    """Applies morphological operations to clean a binary mask."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_open = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = binary_fill_holes(mask_closed).astype(np.uint8)
    return mask_cleaned

@timeit
def select_points_with_spread_with_shuffle(heatmap, num_points, distance_threshold, heatmap_threshold):
    """Randomly selects spread-out points from a heatmap above a threshold."""
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

def select_low_gradient_high_thresh_points_limited(heatmap, gradient_threshold=0.01, value_threshold=0.95, grid_size=(25, 25), max_points_per_grid=5):
    """Selects low-gradient, high-value points with limits per grid."""
    smooth_heatmap = filters.gaussian(heatmap, sigma=2)
    strong_binary_map = smooth_heatmap > value_threshold
    gradient_y, gradient_x = np.gradient(smooth_heatmap)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    low_gradient_map = gradient_magnitude < gradient_threshold
    combined_map = low_gradient_map & strong_binary_map
    labeled_regions, _ = label(combined_map)
    combined_regions = measure.regionprops(labeled_regions, intensity_image=smooth_heatmap)
    centroids = np.array([region.centroid for region in combined_regions], dtype=int)
    grid_height, grid_width = grid_size
    grid_rows = np.floor(centroids[:, 0] / grid_height).astype(int)
    grid_cols = np.floor(centroids[:, 1] / grid_width).astype(int)
    grid_keys = np.column_stack((grid_rows, grid_cols))
    unique_keys, inverse_indices = np.unique(grid_keys, axis=0, return_inverse=True)
    heatmap_values = smooth_heatmap[centroids[:, 0], centroids[:, 1]]
    sorted_indices = np.argsort(-heatmap_values)
    final_selected_points = []
    points_per_grid = np.zeros(len(unique_keys), dtype=int)
    
    for idx in sorted_indices:
        grid_idx = inverse_indices[idx]
        if points_per_grid[grid_idx] < max_points_per_grid:
            final_selected_points.append(centroids[idx])
            points_per_grid[grid_idx] += 1
        if np.all(points_per_grid >= max_points_per_grid):
            break
    
    return np.array(final_selected_points)

@timeit
def process_image_with_clipseg(image, prompts, processor, model, num_points, distance_threshold, heatmap_threshold, selection_method):
    """Processes an image using CLIPSeg and selects points with the chosen method."""
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
        
        if selection_method == 'gradient':
            top_coords = select_low_gradient_high_thresh_points_limited(heatmap_resized, value_threshold=heatmap_threshold)
        elif selection_method == 'random':
            top_coords = select_points_with_spread_with_shuffle(heatmap_resized, num_points, distance_threshold, heatmap_threshold)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")

        if len(top_coords) == 0:
            print(f"Warning: No coordinates found for {prompt}")
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
def plot_heatmap_and_save(image, heatmap_resized, top_coords, prompt, index, orig_size, save_path, mask):
    """Plots and saves the heatmap, original image, and RGB mask with transparent background and text directly at the image edges."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    ax[0].imshow(heatmap_resized, cmap='hot', interpolation='nearest')
    ax[0].scatter([c[1] for c in top_coords], [c[0] for c in top_coords], color='blue')
    ax[0].axis('off')
    ax[0].text(0.01, 0.99, f"Prompt: {prompt}", fontsize=12, ha='left', va='top', transform=ax[0].transAxes,
               bbox=dict(facecolor='white', edgecolor='none', pad=2))
    ax[0].text(0.01, 0.01, "Heatmap", fontsize=10, ha='left', va='bottom', transform=ax[0].transAxes,
               bbox=dict(facecolor='white', edgecolor='none', pad=2))

    ax[1].imshow(image.resize(orig_size))
    ax[1].scatter([c[1] for c in top_coords], [c[0] for c in top_coords], color='blue')
    ax[1].axis('off')
    ax[1].text(0.01, 0.01, "Image", fontsize=10, ha='left', va='bottom', transform=ax[1].transAxes,
               bbox=dict(facecolor='white', edgecolor='none', pad=2))

    ax[2].imshow(mask, cmap='gray')
    ax[2].axis('off')
    ax[2].text(0.01, 0.01, "Segmented", fontsize=10, ha='left', va='bottom', transform=ax[2].transAxes,
               bbox=dict(facecolor='white', edgecolor='none', pad=2))

    plt.subplots_adjust(left=0.01, right=0.99, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

def get_mask_coordinates(mask):
    """
    Get coordinates of all regions in the binary mask.
    
    Args:
        mask (np.array): Binary mask.
    
    Returns:
        list: List of coordinates for each region.
    """
    coords = np.column_stack(np.where(mask == 1))
    return [tuple(coord) for coord in coords]
def create_white_image_with_black_mask(image_size, mask_coords):
    """
    Create a white image and draw the mask coordinates in black.
    
    Args:
        image_size (tuple): Size of the image (width, height).
        mask_coords (list): List of coordinates in the mask.
    
    Returns:
        Image: Image with mask drawn.
    """
    # Create a white image
    white_image = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(white_image)
    
    # Draw the mask coordinates in black
    for coord in mask_coords:
        draw.point(coord, fill='black')
    
    return white_image

@timeit
def apply_sam_and_overlay_masks(image_np, all_coords, prompts, sam_predictor, colors, kernel_size):
    """Applies SAM for segmentation and overlays masks on the image, keeping non-masked areas unchanged."""
    overlay = np.zeros_like(image_np)
    h, w, _ = image_np.shape
    num_prompts = len(prompts)

    if len(colors) < num_prompts:
        raise ValueError(f"Not enough colors provided. Expected {num_prompts}, but got {len(colors)}.")

    colors = colors[:num_prompts]

    combined_mask = np.zeros((h, w, num_prompts), dtype=np.uint8)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    binary_masks = {}
    red_overlays = {}
    image_nps = {}
    for i, (prompt, prompt_coords) in enumerate(zip(prompts, all_coords)):
        if len(prompt_coords) == 0:
            continue 
        
        input_prompts = {
            "point_coords": [(point[1], point[0]) for point in prompt_coords],
            "point_labels": [1] * len(prompt_coords)
        }
        
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            masks, scores, _ = sam_predictor.predict(**input_prompts)
            
        combined_mask[:, :, i] = np.bitwise_or.reduce(masks.astype(np.uint8), axis=0)
        combined_mask[:, :, i] = apply_morphology(combined_mask[:, :, i], kernel_size)
        
        red_overlay = np.zeros_like(image_np)
        red_overlay[:, :, 0] = 255
        binary_masks[prompt] = np.where(combined_mask[:, :, i][:, :, np.newaxis], 
                                        (0.5 * red_overlay + 0.5 * image_np).astype(np.uint8), 
                                        0.5 * image_np).astype(np.uint8)
        red_overlays[prompt] = red_overlay
    
    # Get coordinates of the mask region
    mask_coords = get_mask_coordinates(combined_mask[:, :, i])
    # print(f"Coordinates of the mask region for prompt '{prompt}': {mask_coords}")
    white_image_with_black_mask = create_white_image_with_black_mask((w, h), mask_coords)
     
    # show_combined_mask(combined_mask)

    mask_stack = np.repeat(combined_mask[:, :, :, np.newaxis], 3, axis=3)
    color_array = np.array(colors).reshape(1, 1, num_prompts, 3)
    
    colored_masks = mask_stack * color_array
    overlay = np.max(colored_masks, axis=2).astype(np.uint8)

    final_overlay = (0.5 * image_np + 0.5 * overlay).astype(np.uint8)
    final_mask = np.argmax(combined_mask, axis=2) + 1

    return final_overlay, final_mask, binary_masks,red_overlays,combined_mask

@timeit
def plot_and_save_final_image(image_np, final_image, save_dir):
    """Plots and saves the final segmented image."""
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
    """Loads default RGB colors from a file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            colors = [line.strip() for line in f if line.strip()]
        return ';'.join(colors)

def main(args):
    """Main function to run CLIPSeg and SAM with prompt-based segmentation."""
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
        args.distance_threshold, args.heatmap_threshold, args.selection_method
    )

    final_image, final_mask, binary_masks,red_overlays,combined_mask = apply_sam_and_overlay_masks(
        image_np, all_coords, prompts, sam_predictor, colors, args.kernel_size
    )
    # print(len(binary_masks['road']))

    for result in results:
        # print(f"Processing image: {result['top_coords']}")
        prompt = result['prompt']
        save_path = os.path.join(args.save_dir, f"output_{prompt}_{image_name}.png")
        plot_heatmap_and_save(
            result['image'], result['heatmap_resized'], result['top_coords'], 
            result['prompt'], result['index'], result['orig_size'], save_path, binary_masks[prompt]
        )

    heatmap_log = np.exp(result['heatmap_resized'])
    heatmap_normalized = (heatmap_log - np.min(heatmap_log)) / (np.max(heatmap_log) - np.min(heatmap_log))
    
    Image.fromarray(final_image).save(os.path.join(args.save_dir, args.image_path.split(os.sep)[-1].split('.')[0] + "_segmented.png"))
    Image.fromarray(final_mask.astype(np.uint8)).save(os.path.join(args.save_dir, args.image_path.split(os.sep)[-1].split('.')[0] + "_mask.png"))
    Image.fromarray((heatmap_normalized * 255).astype(np.uint8)).save(os.path.join(args.save_dir, args.image_path.split(os.sep)[-1].split('.')[0] + "_heatmap.png"))
    return results,final_image,binary_masks,red_overlays,combined_mask
def detect(args):
    
    class Params:
        def __init__(self, image_path, save_dir, prompts):
            self.image_path = image_path
            self.save_dir = save_dir
            self.prompts = prompts
            self.seed = 42
            self.num_points = 40
            self.distance_threshold = 35
            self.colors = load_default_colors()
            self.kernel_size = 3
            self.heatmap_threshold = 0.8
            self.selection_method = 'gradient'
    params = Params(args["image_path"], args["save_dir"], args["prompts"])
      
    results,final_image,binary_masks,red_overlays,combined_mask = main(params)
    return results,final_image,binary_masks,red_overlays,combined_mask
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--prompts', type=str, required=True, help='Comma-separated list of prompts for segmentation')
    parser.add_argument('--num_points', type=int, default=40, help='Number of points to select for each prompt')
    parser.add_argument('--distance_threshold', type=int, default=35, help='Minimum distance between selected points')
    parser.add_argument('--colors', type=str, default=load_default_colors(), help='Semicolon-separated RGB colors for each prompt in the format R,G,B')
    parser.add_argument('--save_dir', type=str, default='./saved_figures', help='Directory to save the output images')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for morphological operations to clean masks')
    parser.add_argument('--heatmap_threshold', type=float, default=0.8, help='Threshold for selecting high-value heatmap points')
    parser.add_argument('--selection_method', type=str, default='gradient', choices=['gradient', 'random'], help='Method to select points: gradient or random')
    parser.add_argument('--seed', type=int, default=24, help='Seed for random operations to ensure reproducibility')
    args = parser.parse_args()
    main(args)
