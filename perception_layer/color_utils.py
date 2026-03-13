import warnings
from colorsys import rgb_to_hsv
from itertools import combinations

import numpy as np
from sklearn.cluster import KMeans


def quantize_colors(pil_img, k=3, resize=120):
    """
    Extract dominant colors using K-Means clustering.

    Parameters
    ----------
    pil_img : PIL.Image
        Cropped clothing image.
    k : int
        Number of dominant colors to extract.
    resize : int
        Resize dimension for speed.

    Returns
    -------
    list of tuples
        [(hex_color, percentage), ...]
    """

    img = pil_img.copy()
    img.thumbnail((resize, resize))

    img_data = np.array(img)

    # Handle Grayscale → RGB
    if len(img_data.shape) == 2:
        img_data = np.stack([img_data] * 3, axis=-1)

    # Handle RGBA → RGB
    if img_data.shape[-1] == 4:
        img_data = img_data[:, :, :3]

    pixels = img_data.reshape(-1, 3)

    # ---------------------------------------------------------
    # FIX: Check how many actual unique colors exist in the crop
    # ---------------------------------------------------------
    unique_colors = np.unique(pixels, axis=0)
    actual_k = min(k, len(unique_colors))

    # Edge Case: Completely empty/transparent image somehow slipped through
    if actual_k == 0:
        return [("#FFFFFF", 1.0)]

    # Edge Case: The image is exactly one solid color (no need to run K-Means)
    if actual_k == 1:
        rgb = tuple(unique_colors[0])
        hex_code = '#{:02X}{:02X}{:02X}'.format(*rgb)
        return [(hex_code, 1.0)]

    # Suppress any lingering sklearn warnings and run K-Means with safe cluster count
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=actual_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(pixels)

    centers = kmeans.cluster_centers_.astype(int)

    counts = np.bincount(labels)
    percentages = counts / counts.sum()

    sorted_idx = np.argsort(percentages)[::-1]

    dominant_colors = []
    for i in sorted_idx:
        rgb = tuple(centers[i])
        hex_code = '#{:02X}{:02X}{:02X}'.format(*rgb)
        pct = float(percentages[i])
        dominant_colors.append((hex_code, round(pct, 4)))

    return dominant_colors


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def hue_from_hex(hex_color):
    r, g, b = hex_to_rgb(hex_color)
    h, s, v = rgb_to_hsv(r, g, b)
    return h * 360


def angle_distance(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def harmony_score(color_data):
    """
    color_data: [(hex_color, percentage), ...]

    returns: float between 0 and 1
    """

    hues = [hue_from_hex(hex) for hex, _ in color_data]
    weights = [pct for _, pct in color_data]

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    if len(hues) <= 1:
        return 1.0

    # ideal harmony angles in degrees
    ideal_angles = [30, 60, 90, 120, 150, 180]

    weighted_score = 0
    weight_sum = 0

    for (i, j) in combinations(range(len(hues)), 2):
        d = angle_distance(hues[i], hues[j])

        # find the closest harmonic angle
        closest = min(abs(d - angle) for angle in ideal_angles)

        # convert to score (closer = better)
        pair_score = max(0, 1 - closest / 90)

        pair_weight = weights[i] * weights[j]

        weighted_score += pair_score * pair_weight
        weight_sum += pair_weight

    base_score = weighted_score / weight_sum if weight_sum else 0

    # penalty for chaotic spread (too uneven spacing)
    sorted_hues = sorted(hues)
    gaps = [angle_distance(sorted_hues[i], sorted_hues[(i + 1) % len(sorted_hues)])
            for i in range(len(sorted_hues))]

    gap_variation = max(gaps) - min(gaps)
    penalty = min(gap_variation / 180, 0.3)

    final_score = max(0, min(1, base_score - penalty))

    return round(final_score, 3)


def harmony_score_from_images(images, k=3):
    """
    Compute overall color harmony score across multiple garments.

    Parameters
    ----------
    images : list[PIL.Image]
        Cropped clothing item images.
    k : int
        Number of dominant colors per garment.

    Returns
    -------
    float
        Harmony score between 0 and 1.
    """

    if not images:
        return 0.0

    combined_colors = []

    for img in images:
        color_data = quantize_colors(img, k=k)

        # weight colors by image area (larger garments influence more)
        weight_factor = img.size[0] * img.size[1]

        for hex_color, pct in color_data:
            combined_colors.append((hex_color, pct * weight_factor))

    # normalize combined weights
    total = sum(weight for _, weight in combined_colors)
    combined_colors = [(hex_color, weight / total) for hex_color, weight in combined_colors]

    return harmony_score(combined_colors)


if __name__ == '__main__':
    # Example usage
    from PIL import Image
    import os

    # Just a small sanity check so the example doesn't crash if files are missing
    if os.path.exists("shirt.jpg") and os.path.exists("pants.jpg") and os.path.exists("shoes.jpg"):
        shirt = Image.open("shirt.jpg")
        pants = Image.open("pants.jpg")
        shoes = Image.open("shoes.jpg")

        score = harmony_score_from_images([shirt, pants, shoes])
        print("Outfit harmony:", score)
    else:
        print("Example images not found. Color Utils ready for import.")