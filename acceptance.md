# Acceptance Criteria

(Updated before each feature implementation. Define what "done" means for each task.)

## Task 1: Image Type Validation

### Acceptance Criteria
- [x] A 2D grayscale image (shape (H, W), dtype float32/float64) passes validation
- [x] A 2D color image (shape (H, W, C) where C=3 or 4, dtype float32/float64) passes validation
- [x] A 3D grayscale image (shape (Z, H, W), dtype float32/float64) passes validation
- [x] A 3D color image (shape (Z, H, W, C), dtype float32/float64) passes validation
- [x] A 2D binary image (shape (H, W), dtype bool) passes validation
- [x] A 2D int label image (shape (H, W), dtype int32) passes validation
- [x] An image with wrong dimensions raises a validation error with descriptive message
- [x] An image with wrong dtype raises a validation error with descriptive message
- [x] Validation function returns the input unchanged if valid
- [x] Custom validation error class provides clear error messages

## Task 2: Segmentation Format Conversion

### Acceptance Criteria
- [x] Convert an empty 2D dense label matrix (all zeros) to sparse format, returning an empty sparse array
- [x] Convert a 2D dense matrix with non-overlapping labels to sparse format, preserving all label coordinates
- [x] Convert a 2D dense matrix with overlapping labels (in label_idx dimension) to sparse format correctly
- [x] Convert dense to IJV format (i=row, j=col, v=label value) as a simple coordinate list
- [x] Convert IJV format back to dense labels for a 2D image
- [x] Extract unique label indices from a dense matrix for each slice of label_idx dimension
- [x] Compute area (pixel count) for each label from IJV format
- [x] Count the number of unique labels in an IJV array
- [x] Downsample labels to smallest integer dtype that can hold all values
- [x] Validate dense matrices have correct 6D shape (label_idx, c, t, z, y, x)

## Task 3: Edge Enhancement Functions

### Acceptance Criteria
- [x] Sobel filter detects edges in all directions, returning gradient magnitude image
- [x] Sobel filter can be configured to detect only horizontal or only vertical edges
- [x] Prewitt filter detects edges similar to Sobel but with different kernel weights
- [x] Laplacian of Gaussian (LoG) filter detects edges with configurable sigma for smoothing
- [x] Canny edge detector produces binary edge map with automatic threshold selection
- [x] Canny edge detector supports manual low/high thresholds
- [x] All filters accept optional mask to limit computation to specific regions
- [x] Filter outputs have same shape as input images
- [x] Filters handle float32 and float64 input images

## Task 4: Morphological Operations

### Acceptance Criteria
- [x] Erosion shrinks bright regions / expands dark regions using a structuring element
- [x] Dilation expands bright regions / shrinks dark regions using a structuring element
- [x] Opening (erosion then dilation) removes small bright spots and smooths object boundaries
- [x] Closing (dilation then erosion) fills small holes and connects nearby objects
- [x] Skeletonization reduces binary objects to single-pixel-wide representations
- [x] Morphological gradient computes difference between dilation and erosion
- [x] All operations support disk, square, and diamond structuring element shapes
- [x] Operations work on both 2D and 3D images with appropriate structuring elements
- [x] Structuring element size is configurable

## Task 5: Color Conversion Functions

### Acceptance Criteria
- [x] Convert RGB image to grayscale using luminance weights
- [x] Convert RGBA image to RGB by dropping alpha channel
- [x] Convert RGBA image to grayscale
- [x] Split RGB image into separate R, G, B channels
- [x] Combine separate R, G, B channels into RGB image
- [x] Convert grayscale to RGB (replicate channel 3 times)
- [x] Stretch/normalize channel intensities to 0-1 range
- [x] Handle both float and integer input images

## Task 6: Image Overlap Measurement Functions

### Acceptance Criteria
- [x] Compute true positive, false positive, false negative counts from test vs ground truth labels
- [x] Compute precision (TP / (TP + FP)) for segmentation quality assessment
- [x] Compute recall (TP / (TP + FN)) for segmentation completeness
- [x] Compute F-score (harmonic mean of precision and recall) with configurable beta parameter
- [x] Compute Rand index measuring agreement between two labelings
- [x] Compute adjusted Rand index correcting for chance agreement
- [x] Compute Jaccard index (intersection over union) for each labeled object
- [x] Handle empty labelings gracefully (no objects in test or ground truth)
- [x] Support comparison of 2D and 3D label images

## Task 7: Object Size and Shape Measurement Functions

### Acceptance Criteria
- [x] Compute area (pixel count) for each labeled object
- [x] Compute perimeter (boundary length) for each labeled object
- [x] Compute centroid (center of mass) coordinates for each object
- [x] Compute bounding box (min/max row/col) for each object
- [x] Compute eccentricity measuring how elongated each object is (0 = circle, 1 = line)
- [x] Compute form factor (4*pi*area/perimeter^2) measuring circularity
- [x] Compute major and minor axis lengths from the object's fitted ellipse
- [x] Compute solidity (area / convex hull area) measuring how convex each object is
- [x] Return measurements as dictionaries keyed by object label
- [x] Support both 2D and 3D labeled images
