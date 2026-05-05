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
- [ ] Convert an empty 2D dense label matrix (all zeros) to sparse format, returning an empty sparse array
- [ ] Convert a 2D dense matrix with non-overlapping labels to sparse format, preserving all label coordinates
- [ ] Convert a 2D dense matrix with overlapping labels (in label_idx dimension) to sparse format correctly
- [ ] Convert dense to IJV format (i=row, j=col, v=label value) as a simple coordinate list
- [ ] Convert IJV format back to dense labels for a 2D image
- [ ] Extract unique label indices from a dense matrix for each slice of label_idx dimension
- [ ] Compute area (pixel count) for each label from IJV format
- [ ] Count the number of unique labels in an IJV array
- [ ] Downsample labels to smallest integer dtype that can hold all values
- [ ] Validate dense matrices have correct 6D shape (label_idx, c, t, z, y, x)
