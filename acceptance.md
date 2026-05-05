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
- [ ] Erosion shrinks bright regions / expands dark regions using a structuring element
- [ ] Dilation expands bright regions / shrinks dark regions using a structuring element
- [ ] Opening (erosion then dilation) removes small bright spots and smooths object boundaries
- [ ] Closing (dilation then erosion) fills small holes and connects nearby objects
- [ ] Skeletonization reduces binary objects to single-pixel-wide representations
- [ ] Morphological gradient computes difference between dilation and erosion
- [ ] All operations support disk, square, and diamond structuring element shapes
- [ ] Operations work on both 2D and 3D images with appropriate structuring elements
- [ ] Structuring element size is configurable
