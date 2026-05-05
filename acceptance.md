# Acceptance Criteria

(Updated before each feature implementation. Define what "done" means for each task.)

## Task 1: Image Type Validation

### Acceptance Criteria
- [ ] A 2D grayscale image (shape (H, W), dtype float32/float64) passes validation
- [ ] A 2D color image (shape (H, W, C) where C=3 or 4, dtype float32/float64) passes validation
- [ ] A 3D grayscale image (shape (Z, H, W), dtype float32/float64) passes validation
- [ ] A 3D color image (shape (Z, H, W, C), dtype float32/float64) passes validation
- [ ] A 2D binary image (shape (H, W), dtype bool) passes validation
- [ ] A 2D int label image (shape (H, W), dtype int32) passes validation
- [ ] An image with wrong dimensions raises a validation error with descriptive message
- [ ] An image with wrong dtype raises a validation error with descriptive message
- [ ] Validation function returns the input unchanged if valid
- [ ] Custom validation error class provides clear error messages
