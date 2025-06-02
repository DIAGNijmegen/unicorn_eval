import numpy as np
import collections
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import binary_erosion, label
import itertools
import SimpleITK as sitk
from skimage.measure import regionprops


def calculate_angle_between_lines(point1, point2, point3, point4):
    # Convert points to vectors
    vector1 = np.array([point2[0] - point1[0], point2[1] - point1[1]])
    vector2 = np.array([point4[0] - point3[0], point4[1] - point3[1]])

    # Calculate the dot product of the vectors
    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the cosine of the angle between the vectors
    cos_angle = dot_product / (magnitude1 * magnitude2)

    # Ensure the cosine value is within valid range [-1, 1] for arccos
    cos_angle = np.clip(cos_angle, -1, 1)

    # Calculate the angle in radians
    angle = np.arccos(cos_angle)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle)

    return angle_degrees


def find_perpendicular_diameter(point1, point2, boundary_points):
    max_distance = 0
    short_axis_points = []

    # Get all pair combinations
    bp_combinations = list(itertools.combinations(boundary_points, 2))

    angle_dev = 0
    while len(short_axis_points) == 0:
        for point3, point4 in bp_combinations:
            angle = calculate_angle_between_lines(point1, point2, point3, point4)
            # Check if the angle of the line formed by p1 and p2 is close to perpendicular
            if abs(angle - 90) < angle_dev:
                distance = np.linalg.norm(point3 - point4)
                if distance > max_distance:
                    max_distance = distance
                    short_axis_points = [point3, point4]
        angle_dev += 1

    return max_distance, short_axis_points[0], short_axis_points[1]


def sape(y_true, y_pred):
    """
    Calculates the symmetric absolute percentage error between two measurements
    """
    denominator = abs(y_true) + abs(y_pred)
    if denominator == 0:
        return 0  # Return 0 if both y_true and y_pred are 0
    else:
        return abs(y_pred - y_true) / denominator


def long_and_short_axis_diameters(mask):
    """
    Calculates the long- and short-axis diameters of a lesion from the 3D segmentation mask
    by fitting an ellipse to each axial component and taking the largest major/minor lengths.
    Returns:
      long_axis_diameter, short_axis_diameter,
      long_axis_points, short_axis_points
    (Note: points are the centroids ± half‐axes in pixel coords; optional.)
    """
    best_long, best_short = 0.0, 0.0
    best_long_pts = None
    best_short_pts = None

    for z, slice_2d in enumerate(mask):
        if slice_2d.max() == 0:
            continue

        # keep only the largest connected component
        lab, n = label(slice_2d)
        if n > 1:
            counts = collections.Counter(lab.flat)
            bg, _ = counts.most_common(1)[0]
            lab = np.where(lab == bg, 1, 0)
        else:
            lab = (lab > 0).astype(int)

        # use regionprops to fit ellipse & get axis lengths
        props = regionprops(lab)[0]
        long_diam  = props.major_axis_length
        short_diam = props.minor_axis_length

        # optional: compute the two endpoints along each axis in pixel coords
        cy, cx = props.centroid
        orientation = props.orientation  # radians CCW from the horizontal
        dy = np.sin(orientation) * (long_diam/2)
        dx = np.cos(orientation) * (long_diam/2)
        long_pts = [(cx - dx, cy - dy, z), (cx + dx, cy + dy, z)]

        dy2 = np.cos(orientation) * (short_diam/2)
        dx2 = -np.sin(orientation) * (short_diam/2)
        short_pts = [(cx - dx2, cy - dy2, z), (cx + dx2, cy + dy2, z)]

        # keep the slice with the maximum long axis
        if long_diam > best_long:
            best_long   = long_diam
            best_short  = short_diam
            best_long_pts  = long_pts
            best_short_pts = short_pts

    return best_long, best_short, best_long_pts, best_short_pts



def dice_coefficient(mask1, mask2):
    mask1 = np.asarray(mask1).astype(bool)
    mask2 = np.asarray(mask2).astype(bool)
    # Calculate intersection
    intersection = np.logical_and(mask1, mask2)
    # Calculate Dice
    dice = 2.0 * intersection.sum() / (mask1.sum() + mask2.sum())
    if np.isnan(dice):
        return 0
    else:
        return dice


def compute_uls_score(gts, preds):
    uls_scores = 0
    for i, gt in enumerate(gts):
        pred = preds[i]
        gt_long, gt_short, _, _ = long_and_short_axis_diameters(gt)
        pred_long, pred_short, _, _ = long_and_short_axis_diameters(pred)

        dice = dice_coefficient(gt, pred)
        sape_long = sape(gt_long, pred_long)
        sape_short = sape(gt_short, pred_short)
        uls_score = (
            0.888 * dice
            + 0.056 * (1 - min(1, sape_long))
            + 0.056 * (1 - min(1, sape_short))
        )
        uls_scores += uls_score

    uls_score_final = uls_scores / len(gts)
    return uls_score_final
