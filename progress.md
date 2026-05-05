# Progress

(Updated after each feature commit.)

## Round 1
**Task**: Task 1 — Implement image type validation
**Files created**: cellprofiler_lib/__init__.py, cellprofiler_lib/types.py, tests/test_types.py, pytest.ini
**Commit**: Add image type validation that ensures numpy arrays conform to expected constraints for 2D and 3D images
**Acceptance**: 10/10 criteria met
**Verification**: tests FAIL on previous state (ModuleNotFoundError), PASS on current state

## Round 2
**Task**: Task 2 — Implement segmentation format conversion functions
**Files created**: cellprofiler_lib/segmentation.py, tests/test_segmentation.py
**Commit**: Add segmentation format conversion functions that allow users to convert between different representations of labeled objects
**Acceptance**: 10/10 criteria met
**Verification**: tests FAIL on previous state (ModuleNotFoundError), PASS on current state
