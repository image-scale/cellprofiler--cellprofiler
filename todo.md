# Todo

## Plan
Build the image processing library from the ground up, starting with image type definitions and validation, then implementing segmentation functions for label handling, followed by image processing operations like edge enhancement and morphology. Finally, add measurement functions for image analysis. Each task delivers a complete, testable feature.

## Tasks
- [x] Task 1: Implement image type validation that validates numpy arrays as 2D/3D images with proper shape and dtype constraints, supporting grayscale, color, binary, and mask image types (types.py + tests/test_types.py)
- [x] Task 2: Implement segmentation format conversion functions that convert between dense label matrices (6D arrays), sparse label formats (structured arrays), and IJV coordinate lists, enabling flexible representation of object segmentations (segmentation.py + tests/test_segmentation.py)
- [x] Task 3: Implement edge enhancement functions including Sobel, Prewitt, Canny, and Laplacian of Gaussian filters that detect edges in grayscale images with optional mask support and direction control (edges.py + tests/test_edges.py)
- [x] Task 4: Implement morphological operations including opening, closing, erosion, dilation, and skeletonization that transform binary and grayscale images using configurable structuring elements (morphology.py + tests/test_morphology.py)
- [x] Task 5: Implement color conversion functions that convert between RGB, grayscale, and channel-separated representations, supporting both 3-channel and 4-channel (RGBA) input images (color.py + tests/test_color.py)
- [x] Task 6: Implement image overlap measurement functions that compute precision, recall, F-score, Rand index, and confusion matrix statistics when comparing test segmentations to ground truth (measurement.py + tests/test_measurement.py)
- [>] Task 7: Implement object size and shape measurement functions that compute area, perimeter, centroid, bounding box, eccentricity, and form factor for labeled objects in segmented images (object_measurement.py + tests/test_object_measurement.py)
- [ ] Task 8: Implement image intensity measurement functions that compute min, max, mean, median, standard deviation, and percentile statistics for pixel intensities within object regions (intensity_measurement.py + tests/test_intensity_measurement.py)
- [ ] Task 9: Implement image transformation functions including resize, crop, flip, and rotate operations that modify image dimensions and orientation with configurable interpolation methods (transforms.py + tests/test_transforms.py)
- [ ] Task 10: Implement a pipeline framework that allows users to define analysis workflows as sequences of processing modules, execute them on image data, and collect measurements from each step (pipeline.py + tests/test_pipeline.py)
