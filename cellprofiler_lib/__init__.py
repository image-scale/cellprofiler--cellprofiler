"""
CellProfiler Library - Image analysis functions for cell biology research.
"""
from .types import (
    ImageValidationError,
    validate_image,
    validate_2d_grayscale,
    validate_2d_color,
    validate_3d_grayscale,
    validate_3d_color,
    validate_2d_binary,
    validate_3d_binary,
    validate_2d_labels,
    validate_3d_labels,
    validate_2d_mask,
    validate_3d_mask,
)

from .segmentation import (
    SparseAxis,
    DenseAxis,
    validate_dense_matrix,
    validate_labels,
    validate_ijv,
    validate_sparse,
    indices_from_dense,
    convert_dense_to_sparse,
    convert_labels_to_dense,
    downsample_labels,
    indices_from_ijv,
    count_from_ijv,
    areas_from_ijv,
    convert_ijv_to_sparse,
    convert_sparse_to_ijv,
    convert_labels_to_ijv,
    convert_ijv_to_labels,
    convert_dense_to_labelset,
)

from .edges import (
    EdgeDirection,
    sobel_filter,
    prewitt_filter,
    laplacian_of_gaussian,
    canny_edge_detector,
    compute_gradient_magnitude,
    compute_gradient_direction,
    scharr_filter,
)

from .morphology import (
    StructuringElementShape,
    create_structuring_element,
    erode,
    dilate,
    opening,
    closing,
    skeletonize,
    morphological_gradient,
    white_tophat,
    black_tophat,
    fill_holes,
    remove_small_objects,
    remove_small_holes,
)

from .color import (
    rgb_to_grayscale,
    rgba_to_rgb,
    rgba_to_grayscale,
    split_channels,
    combine_channels,
    grayscale_to_rgb,
    normalize_intensity,
    stretch_intensity,
    invert_image,
    adjust_gamma,
    extract_channel,
    replace_channel,
    to_float,
    to_uint8,
)

from .measurement import (
    OverlapStatistics,
    compute_confusion_counts,
    compute_precision,
    compute_recall,
    compute_f_score,
    compute_rand_index,
    compute_adjusted_rand_index,
    compute_jaccard_index,
    compute_jaccard_per_object,
    compute_overlap_statistics,
    compute_dice_coefficient,
)

from .object_measurement import (
    ObjectMeasurements,
    measure_object_area,
    measure_object_perimeter,
    measure_object_centroid,
    measure_object_bounding_box,
    measure_object_eccentricity,
    measure_object_major_axis_length,
    measure_object_minor_axis_length,
    measure_object_orientation,
    measure_object_solidity,
    measure_object_extent,
    measure_object_form_factor,
    measure_all_object_properties,
    measure_object_equivalent_diameter,
    measure_object_euler_number,
    measure_object_convex_area,
    measure_object_filled_area,
)

from .intensity_measurement import (
    IntensityMeasurements,
    measure_intensity_min,
    measure_intensity_max,
    measure_intensity_mean,
    measure_intensity_median,
    measure_intensity_std,
    measure_intensity_percentile,
    measure_intensity_percentiles,
    measure_integrated_intensity,
    measure_intensity_mad,
    measure_all_intensity_properties,
    measure_intensity_in_channels,
    measure_intensity_ratio,
    measure_intensity_at_edge,
    measure_intensity_at_center,
)

from .transforms import (
    InterpolationOrder,
    PaddingMode,
    resize,
    crop,
    crop_center,
    flip_horizontal,
    flip_vertical,
    flip_depth,
    rotate,
    rotate_90,
    pad,
    translate,
    affine_transform,
    scale,
    transpose,
    rescale_intensity,
)

from .pipeline import (
    ModuleCategory,
    Workspace,
    Module,
    Pipeline,
    ThresholdModule,
    IdentifyObjectsModule,
    MeasureObjectSizeModule,
    MeasureIntensityModule,
    GaussianSmoothModule,
    RescaleIntensityModule,
    InvertModule,
    CropModule,
)

__version__ = "0.1.0"
