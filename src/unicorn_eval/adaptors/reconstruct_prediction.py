from itertools import product

import numpy as np
import SimpleITK as sitk


def _dir_flat_to_mat(direction_flat):
    """direction (len = D*D) -> (D,D) column-major matrix as used by SimpleITK."""
    D = int(round(np.sqrt(len(direction_flat))))
    M = np.asarray(direction_flat, dtype=float).reshape(D, D)
    return M


def _patch_corners_world(start_phys, size_xyz, spacing_xyz, direction_flat):
    """
    Return all physical-space corner points of a patch whose index (0,0,0) is at start_phys.
    size_xyz and spacing_xyz are (x,y[,z]) in SimpleITK convention; direction is D*D flat tuple.
    """
    D = len(size_xyz)
    Dmat = _dir_flat_to_mat(direction_flat)  # columns are axis directions (world basis)
    # axis extents in world space
    ext_cols = [Dmat[:, i] * (size_xyz[i] * spacing_xyz[i]) for i in range(D)]
    corners = []
    for bits in product([0,1], repeat=D):
        p = np.array(start_phys, dtype=float)
        for i, b in enumerate(bits):
            if b:
                p = p + ext_cols[i]
        corners.append(p)
    return corners


def _project_to_dir_coords(points_world, direction_flat):
    """Project world points into the coordinate system defined by 'direction' columns."""
    Dmat = _dir_flat_to_mat(direction_flat)
    # In that basis, coords u satisfy p = D * u  =>  u = D^T * p   (since D is orthonormal)
    DT = Dmat.T
    points_world = np.asarray(points_world, dtype=float)
    return (DT @ points_world.T).T


def stitch_patches(patches, pixel_id=sitk.sitkFloat32, average_overlaps=False):
    """
    patches: list of dicts, each with:
      - 'features': numpy array shaped (z,y,x) for 3D or (y,x) for 2D (SimpleITK convention).
      - 'patch_size'     : (x,y[,z])         # size in index space for this patch
      - 'patch_spacing'  : (sx,sy[,sz])
      - 'image_direction': tuple/list length D*D
      - 'coord'    : (x_phys, y_phys[, z_phys])  # physical location of index (0,0,0) of the patch
    Assumptions:
      * All patches are on the same grid: identical spacing & direction (we enforce/verify).
      * 'coord' is the physical location of the PATCH VOXEL CENTER at index (0,0,0).
        If your 'coord' refers to a corner, add 0.5*patch_spacing in each axis before calling.
    """
    if not patches:
        raise ValueError("No patches provided.")

    # Use the first patch as the reference geometry
    ref_spacing = tuple(map(float, patches[0]['patch_spacing']))
    ref_direction = tuple(map(float, patches[0]['image_direction']))
    D = len(ref_spacing)

    # Verify common spacing & direction
    for i, p in enumerate(patches):
        if tuple(map(float, p['patch_spacing'])) != ref_spacing:
            raise ValueError(f"Patch {i} has different spacing: {p['patch_spacing']} vs {ref_spacing}")
        if tuple(map(float, p['image_direction'])) != ref_direction:
            raise ValueError(f"Patch {i} has different direction.")

    # 1) Compute global bounding box in the reference direction coordinates
    all_corners_world = []
    for p in patches:
        size_xyz = tuple(int(v) for v in p['patch_size'])
        start_world = tuple(float(v) for v in p['coord'])
        corners = _patch_corners_world(start_world, size_xyz, ref_spacing, ref_direction)
        all_corners_world.extend(corners)

    # Project to direction-basis to get per-axis min/max along the image axes
    corners_dircoords = _project_to_dir_coords(all_corners_world, ref_direction)
    min_u = np.min(corners_dircoords, axis=0)
    max_u = np.max(corners_dircoords, axis=0)

    # Compute output origin (in world) and size (in voxels)
    # u = (index * spacing) + u_origin  with index in [0..size-1]
    # Choose u_origin = min_u rounded to the grid
    u_origin = min_u
    size_f = (max_u - u_origin) / np.array(ref_spacing, dtype=float)
    out_size_xyz = tuple(int(np.round(size_f[i])) for i in range(D))

    # Convert u_origin (dir-coords) back to world origin
    Dmat = _dir_flat_to_mat(ref_direction)
    out_origin_world = tuple((Dmat @ u_origin).tolist())

    # 2) Create the output image
    out_img = sitk.Image(list(out_size_xyz), pixel_id)
    out_img.SetSpacing(ref_spacing)
    out_img.SetDirection(ref_direction)
    out_img.SetOrigin(out_origin_world)

    if average_overlaps:
        sum_img = sitk.Image(out_img)  # same geometry
        count_img = sitk.Image(out_img)  # will hold counts (as float)
        sum_img = sitk.Cast(sum_img, sitk.sitkFloat64)
        count_img = sitk.Cast(count_img, sitk.sitkFloat64)

    # 3) Paste each patch
    for p in patches:
        arr = p['features']
        if arr.shape[0] == 1 and arr.ndim == 4:
            arr = arr[0]

        # Make a SimpleITK image from the numpy array.
        # GetImageFromArray expects (z,y,x) for 3D / (y,x) for 2D which is what we assume here.
        src = sitk.GetImageFromArray(arr)
        src = sitk.Cast(src, pixel_id)
        src.SetSpacing(ref_spacing)
        src.SetDirection(ref_direction)
        src.SetOrigin(tuple(float(v) for v in p['coord']))  # index (0,0,0) center in world

        # Where does this patch start in the OUTPUT index space?
        dest_index = out_img.TransformPhysicalPointToIndex(src.GetOrigin())

        # Paste overwrites values inside the destination region.
        if not average_overlaps:
            out_img = sitk.Paste(out_img, src, src.GetSize(), sourceIndex=(0,)*D, destinationIndex=dest_index)
        else:
            # Accumulate sums
            sum_img = sitk.Paste(sum_img, sitk.Cast(src, sitk.sitkFloat64), src.GetSize(),
                                 sourceIndex=(0,)*D, destinationIndex=dest_index)
            # Accumulate counts (paste an image of ones matching the src size)
            ones = sitk.Image(src.GetSize(), sitk.sitkFloat64)
            ones.SetSpacing(ref_spacing); ones.SetDirection(ref_direction); ones.SetOrigin(src.GetOrigin())
            ones = ones + 1.0
            count_img = sitk.Paste(count_img, ones, ones.GetSize(),
                                   sourceIndex=(0,)*D, destinationIndex=dest_index)

    if average_overlaps:
        # Avoid divide-by-zero: clamp zeros to 1 (regions never filled will stay 0 in sum_img)
        eps = sitk.Cast(count_img == 0, sitk.sitkFloat64) + count_img
        out_img = sitk.Cast(sum_img / eps, pixel_id)

    return out_img


def stitch_patches_fast(patches: list[dict]):
    if not patches:
        raise ValueError("No patches provided.")

    ref_spacing = tuple(map(float, patches[0]['patch_spacing']))
    ref_direction = tuple(map(float, patches[0]['image_direction']))
    D = len(ref_spacing)

    # Validate patches
    for i, p in enumerate(patches):
        if tuple(map(float, p['patch_spacing'])) != ref_spacing:
            raise ValueError(f"Patch {i} has different spacing: {p['patch_spacing']} vs {ref_spacing}")
        if tuple(map(float, p['image_direction'])) != ref_direction:
            raise ValueError(f"Patch {i} has different direction.")

    # Compute bounding box in index space
    all_corners_world = []
    for p in patches:
        size_xyz = tuple(int(v) for v in p['patch_size'])
        start_world = tuple(float(v) for v in p['coord'])
        corners = _patch_corners_world(start_world, size_xyz, ref_spacing, ref_direction)
        all_corners_world.extend(corners)

    corners_dircoords = _project_to_dir_coords(all_corners_world, ref_direction)
    min_u = np.min(corners_dircoords, axis=0)
    max_u = np.max(corners_dircoords, axis=0)

    u_origin = min_u
    size_f = (max_u - u_origin) / np.array(ref_spacing, dtype=float)
    out_size_xyz = tuple(int(np.round(size_f[i])) for i in range(D))

    Dmat = _dir_flat_to_mat(ref_direction)
    out_origin_world = tuple((Dmat @ u_origin).tolist())

    # Allocate arrays
    sum_arr = np.zeros(out_size_xyz[::-1], dtype=np.float64)  # (z,y,x)
    count_arr = np.zeros_like(sum_arr)

    # Accumulate patches
    for p in patches:
        arr = np.asarray(p["features"], dtype=np.float64)
        if arr.ndim != 3:
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]  # drop class dimension
            else:
                raise ValueError("Expected arr shape (1, z, y, x)")

        src = sitk.GetImageFromArray(arr)
        src.SetSpacing(ref_spacing)
        src.SetDirection(ref_direction)
        src.SetOrigin(tuple(float(v) for v in p["coord"]))

        dest_index = sitk.Image(out_size_xyz, sitk.sitkUInt8)
        dest_index.SetSpacing(ref_spacing)
        dest_index.SetDirection(ref_direction)
        dest_index.SetOrigin(out_origin_world)
        dest_idx = dest_index.TransformPhysicalPointToIndex(src.GetOrigin())

        # Compute numpy slices
        slices = tuple(slice(dest_idx[::-1][d], dest_idx[::-1][d] + arr.shape[d]) for d in range(D))

        sum_arr[slices] += arr
        count_arr[slices] += 1

    # Average prediction in case of overlaps
    with np.errstate(divide="ignore", invalid="ignore"):
        out_arr = np.true_divide(sum_arr, count_arr, where=(count_arr > 0))

    # Convert back to SimpleITK
    out_img = sitk.GetImageFromArray(out_arr.astype(np.float32))
    out_img.SetSpacing(ref_spacing)
    out_img.SetDirection(ref_direction)
    out_img.SetOrigin(out_origin_world)

    return out_img
