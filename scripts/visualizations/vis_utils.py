import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random



def get_new_pallete(num_colors, return_type='uint8'):
    """Generate a color pallete given the number of colors needed. First color is always black."""
    pallete = []
    for j in range(num_colors):
        lab = j
        r, g, b = 0, 0, 0
        i = 0
        while lab > 0:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
        pallete.append([r, g, b])

    if return_type == 'uint8':
        return np.array(pallete).astype(np.uint8)
    return np.array(pallete).astype(np.float32) / 255.0


def vis_id_map(id_map, save_path='vis_id_map.png', draw_contours=True):
    """
    Visualize an ID map as a color-coded image.
    """
    unique_ids = np.unique(id_map)
    print(f"Unique IDs in the map: {unique_ids}")
    id_color_map = get_new_pallete(max(unique_ids)+1)
    height, width = id_map.shape
    colored_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for label_id in unique_ids:
        if label_id == 0:
            continue
        id_mask = (id_map == label_id)
        colored_img[id_mask] = id_color_map[label_id]
        if draw_contours:
            contours, _ = cv2.findContours(
                id_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(colored_img, contours, -1, (255,255,255,0.4))
    
    cv2.imwrite(save_path, cv2.cvtColor(colored_img, cv2.COLOR_RGB2BGR))

def vis_depth_map(depth_map, save_path='vis_depth_map.png'):
    """
    Visualize a depth map.
    """
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    print(f"Depth min: {depth_min}, max: {depth_max}")
    depth_vis = (depth_map - depth_min) / (depth_max - depth_min + 1e-8) * 255.0
    depth_vis = depth_vis.astype(np.uint8)
    cv2.imwrite(save_path, depth_vis)

def vis_normal_map(normal_map, save_path='vis_normal_map.png'):
    """
    Visualize a normal map.
    """
    normal_vis = ((normal_map + 1.0) / 2.0 * 255.0).astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))

def main():
    label_image_path = './scene0001_00/2d-instance-filt/0.png'
    # Load the label image
    label_img = np.array(Image.open(label_image_path))
    
    vis_id_map(label_img, save_path='vis_label_image.png')

if __name__ == "__main__":
    main()