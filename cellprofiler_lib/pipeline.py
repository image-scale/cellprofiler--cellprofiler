"""
Pipeline framework for defining and executing image analysis workflows.

This module provides a framework for building image analysis pipelines
as sequences of processing modules that can be executed on image data.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
from numpy.typing import NDArray


class ModuleCategory(Enum):
    """Categories of pipeline modules."""
    IMAGE_PROCESSING = "image_processing"
    SEGMENTATION = "segmentation"
    MEASUREMENT = "measurement"
    OBJECT_PROCESSING = "object_processing"
    DATA_EXPORT = "data_export"


@dataclass
class Workspace:
    """Shared workspace for inter-module communication.

    The workspace holds images, objects, and measurements that modules
    can read from and write to during pipeline execution.
    """
    images: Dict[str, NDArray] = field(default_factory=dict)
    objects: Dict[str, NDArray] = field(default_factory=dict)
    measurements: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_image(self, name: str) -> NDArray:
        """Get an image from the workspace."""
        if name not in self.images:
            raise KeyError(f"Image '{name}' not found in workspace")
        return self.images[name]

    def set_image(self, name: str, image: NDArray) -> None:
        """Store an image in the workspace."""
        self.images[name] = image

    def get_objects(self, name: str) -> NDArray:
        """Get a labeled object image from the workspace."""
        if name not in self.objects:
            raise KeyError(f"Objects '{name}' not found in workspace")
        return self.objects[name]

    def set_objects(self, name: str, labels: NDArray) -> None:
        """Store a labeled object image in the workspace."""
        self.objects[name] = labels

    def add_measurement(
        self,
        object_name: str,
        measurement_name: str,
        values: Any,
    ) -> None:
        """Add a measurement to the workspace."""
        if object_name not in self.measurements:
            self.measurements[object_name] = {}
        self.measurements[object_name][measurement_name] = values

    def get_measurement(
        self,
        object_name: str,
        measurement_name: str,
    ) -> Any:
        """Get a measurement from the workspace."""
        if object_name not in self.measurements:
            raise KeyError(f"Object '{object_name}' has no measurements")
        if measurement_name not in self.measurements[object_name]:
            raise KeyError(
                f"Measurement '{measurement_name}' not found for '{object_name}'"
            )
        return self.measurements[object_name][measurement_name]

    def clear(self) -> None:
        """Clear all workspace data."""
        self.images.clear()
        self.objects.clear()
        self.measurements.clear()
        self.metadata.clear()


class Module(ABC):
    """Base class for pipeline modules.

    All processing steps in a pipeline must inherit from this class
    and implement the run() method.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the module.

        Parameters
        ----------
        name : str, optional
            Name of this module instance. Defaults to class name.
        """
        self._name = name or self.__class__.__name__
        self._enabled = True

    @property
    def name(self) -> str:
        """Get the module name."""
        return self._name

    @property
    def enabled(self) -> bool:
        """Check if module is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the module."""
        self._enabled = value

    @property
    @abstractmethod
    def category(self) -> ModuleCategory:
        """Get the module category."""
        pass

    @abstractmethod
    def run(self, workspace: Workspace) -> None:
        """Execute the module on the workspace.

        Parameters
        ----------
        workspace : Workspace
            Shared workspace containing images and measurements.
        """
        pass

    def validate_settings(self) -> List[str]:
        """Validate module settings.

        Returns
        -------
        List[str]
            List of validation error messages. Empty if valid.
        """
        return []

    def get_measurements(self) -> List[str]:
        """Get names of measurements this module produces.

        Returns
        -------
        List[str]
            List of measurement names.
        """
        return []


class Pipeline:
    """A sequence of modules that process image data.

    The pipeline executes modules in order, passing results through
    a shared workspace.
    """

    def __init__(self, modules: Optional[List[Module]] = None):
        """Initialize the pipeline.

        Parameters
        ----------
        modules : List[Module], optional
            Initial list of modules.
        """
        self._modules: List[Module] = list(modules) if modules else []
        self._workspace = Workspace()

    @property
    def modules(self) -> List[Module]:
        """Get the list of modules."""
        return self._modules.copy()

    @property
    def workspace(self) -> Workspace:
        """Get the workspace."""
        return self._workspace

    def add_module(self, module: Module, index: Optional[int] = None) -> None:
        """Add a module to the pipeline.

        Parameters
        ----------
        module : Module
            Module to add.
        index : int, optional
            Position to insert module. If None, appends to end.
        """
        if index is None:
            self._modules.append(module)
        else:
            self._modules.insert(index, module)

    def remove_module(self, module_or_index: Union[Module, int]) -> Module:
        """Remove a module from the pipeline.

        Parameters
        ----------
        module_or_index : Module or int
            Module instance or index to remove.

        Returns
        -------
        Module
            The removed module.
        """
        if isinstance(module_or_index, int):
            return self._modules.pop(module_or_index)
        else:
            self._modules.remove(module_or_index)
            return module_or_index

    def move_module(self, from_index: int, to_index: int) -> None:
        """Move a module to a new position.

        Parameters
        ----------
        from_index : int
            Current position of module.
        to_index : int
            New position for module.
        """
        module = self._modules.pop(from_index)
        self._modules.insert(to_index, module)

    def validate(self) -> List[str]:
        """Validate all modules in the pipeline.

        Returns
        -------
        List[str]
            List of validation error messages. Empty if all valid.
        """
        errors = []
        for i, module in enumerate(self._modules):
            module_errors = module.validate_settings()
            for error in module_errors:
                errors.append(f"Module {i} ({module.name}): {error}")
        return errors

    def run(
        self,
        image: Optional[NDArray] = None,
        image_name: str = "input",
    ) -> Workspace:
        """Execute the pipeline.

        Parameters
        ----------
        image : NDArray, optional
            Input image to process.
        image_name : str
            Name to assign to input image in workspace.

        Returns
        -------
        Workspace
            Workspace containing all results and measurements.
        """
        # Clear workspace
        self._workspace.clear()

        # Add input image if provided
        if image is not None:
            self._workspace.set_image(image_name, image)

        # Run each enabled module
        for module in self._modules:
            if module.enabled:
                module.run(self._workspace)

        return self._workspace

    def get_all_measurements(self) -> Dict[str, List[str]]:
        """Get all measurements that would be produced by the pipeline.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping module name to list of measurement names.
        """
        result = {}
        for module in self._modules:
            measurements = module.get_measurements()
            if measurements:
                result[module.name] = measurements
        return result


# =============================================================================
# Example Modules
# =============================================================================

class ThresholdModule(Module):
    """Module that applies thresholding to create binary images."""

    def __init__(
        self,
        input_image: str = "input",
        output_image: str = "binary",
        threshold: float = 0.5,
        above: bool = True,
        name: Optional[str] = None,
    ):
        """Initialize threshold module.

        Parameters
        ----------
        input_image : str
            Name of input image in workspace.
        output_image : str
            Name to assign to output binary image.
        threshold : float
            Threshold value.
        above : bool
            If True, pixels above threshold are set to True.
        name : str, optional
            Module name.
        """
        super().__init__(name)
        self.input_image = input_image
        self.output_image = output_image
        self.threshold = threshold
        self.above = above

    @property
    def category(self) -> ModuleCategory:
        return ModuleCategory.IMAGE_PROCESSING

    def run(self, workspace: Workspace) -> None:
        image = workspace.get_image(self.input_image)

        if self.above:
            binary = image > self.threshold
        else:
            binary = image < self.threshold

        workspace.set_image(self.output_image, binary)

    def validate_settings(self) -> List[str]:
        errors = []
        if not 0 <= self.threshold <= 1:
            errors.append(f"Threshold {self.threshold} should be between 0 and 1")
        return errors


class IdentifyObjectsModule(Module):
    """Module that identifies objects in a binary image."""

    def __init__(
        self,
        input_image: str = "binary",
        output_objects: str = "objects",
        min_size: int = 0,
        name: Optional[str] = None,
    ):
        """Initialize identify objects module.

        Parameters
        ----------
        input_image : str
            Name of input binary image in workspace.
        output_objects : str
            Name to assign to output label image.
        min_size : int
            Minimum object size in pixels.
        name : str, optional
            Module name.
        """
        super().__init__(name)
        self.input_image = input_image
        self.output_objects = output_objects
        self.min_size = min_size

    @property
    def category(self) -> ModuleCategory:
        return ModuleCategory.SEGMENTATION

    def run(self, workspace: Workspace) -> None:
        import scipy.ndimage

        binary = workspace.get_image(self.input_image)
        labels, num_objects = scipy.ndimage.label(binary)

        # Remove small objects if min_size specified
        if self.min_size > 0:
            from skimage.morphology import remove_small_objects
            labels = remove_small_objects(labels, min_size=self.min_size)
            # Relabel to ensure consecutive labels
            labels, _ = scipy.ndimage.label(labels > 0)

        workspace.set_objects(self.output_objects, labels)
        workspace.metadata[f"{self.output_objects}_count"] = np.max(labels)


class MeasureObjectSizeModule(Module):
    """Module that measures object sizes."""

    def __init__(
        self,
        input_objects: str = "objects",
        name: Optional[str] = None,
    ):
        """Initialize measure size module.

        Parameters
        ----------
        input_objects : str
            Name of labeled objects in workspace.
        name : str, optional
            Module name.
        """
        super().__init__(name)
        self.input_objects = input_objects

    @property
    def category(self) -> ModuleCategory:
        return ModuleCategory.MEASUREMENT

    def run(self, workspace: Workspace) -> None:
        from skimage.measure import regionprops

        labels = workspace.get_objects(self.input_objects)
        regions = regionprops(labels)

        areas = {int(r.label): int(r.area) for r in regions}
        perimeters = {int(r.label): float(r.perimeter) for r in regions}
        centroids = {int(r.label): tuple(r.centroid) for r in regions}

        workspace.add_measurement(self.input_objects, "area", areas)
        workspace.add_measurement(self.input_objects, "perimeter", perimeters)
        workspace.add_measurement(self.input_objects, "centroid", centroids)

    def get_measurements(self) -> List[str]:
        return ["area", "perimeter", "centroid"]


class MeasureIntensityModule(Module):
    """Module that measures object intensity statistics."""

    def __init__(
        self,
        input_image: str = "input",
        input_objects: str = "objects",
        name: Optional[str] = None,
    ):
        """Initialize measure intensity module.

        Parameters
        ----------
        input_image : str
            Name of intensity image in workspace.
        input_objects : str
            Name of labeled objects in workspace.
        name : str, optional
            Module name.
        """
        super().__init__(name)
        self.input_image = input_image
        self.input_objects = input_objects

    @property
    def category(self) -> ModuleCategory:
        return ModuleCategory.MEASUREMENT

    def run(self, workspace: Workspace) -> None:
        image = workspace.get_image(self.input_image)
        labels = workspace.get_objects(self.input_objects)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]

        means = {}
        stds = {}
        integrated = {}

        for label in unique_labels:
            mask = labels == label
            values = image[mask]
            means[int(label)] = float(np.mean(values))
            stds[int(label)] = float(np.std(values))
            integrated[int(label)] = float(np.sum(values))

        workspace.add_measurement(self.input_objects, "mean_intensity", means)
        workspace.add_measurement(self.input_objects, "std_intensity", stds)
        workspace.add_measurement(self.input_objects, "integrated_intensity", integrated)

    def get_measurements(self) -> List[str]:
        return ["mean_intensity", "std_intensity", "integrated_intensity"]


class GaussianSmoothModule(Module):
    """Module that applies Gaussian smoothing."""

    def __init__(
        self,
        input_image: str = "input",
        output_image: str = "smoothed",
        sigma: float = 1.0,
        name: Optional[str] = None,
    ):
        """Initialize Gaussian smooth module.

        Parameters
        ----------
        input_image : str
            Name of input image in workspace.
        output_image : str
            Name to assign to output image.
        sigma : float
            Standard deviation for Gaussian kernel.
        name : str, optional
            Module name.
        """
        super().__init__(name)
        self.input_image = input_image
        self.output_image = output_image
        self.sigma = sigma

    @property
    def category(self) -> ModuleCategory:
        return ModuleCategory.IMAGE_PROCESSING

    def run(self, workspace: Workspace) -> None:
        from scipy.ndimage import gaussian_filter

        image = workspace.get_image(self.input_image)
        smoothed = gaussian_filter(image, sigma=self.sigma)
        workspace.set_image(self.output_image, smoothed)

    def validate_settings(self) -> List[str]:
        errors = []
        if self.sigma <= 0:
            errors.append(f"Sigma must be positive, got {self.sigma}")
        return errors


class RescaleIntensityModule(Module):
    """Module that rescales image intensity."""

    def __init__(
        self,
        input_image: str = "input",
        output_image: str = "rescaled",
        out_min: float = 0.0,
        out_max: float = 1.0,
        name: Optional[str] = None,
    ):
        """Initialize rescale intensity module.

        Parameters
        ----------
        input_image : str
            Name of input image in workspace.
        output_image : str
            Name to assign to output image.
        out_min : float
            Minimum output value.
        out_max : float
            Maximum output value.
        name : str, optional
            Module name.
        """
        super().__init__(name)
        self.input_image = input_image
        self.output_image = output_image
        self.out_min = out_min
        self.out_max = out_max

    @property
    def category(self) -> ModuleCategory:
        return ModuleCategory.IMAGE_PROCESSING

    def run(self, workspace: Workspace) -> None:
        image = workspace.get_image(self.input_image)
        image = image.astype(np.float64)

        in_min, in_max = image.min(), image.max()

        if in_max == in_min:
            rescaled = np.full_like(image, (self.out_min + self.out_max) / 2)
        else:
            scale = (self.out_max - self.out_min) / (in_max - in_min)
            rescaled = (image - in_min) * scale + self.out_min

        workspace.set_image(self.output_image, rescaled)


class InvertModule(Module):
    """Module that inverts image intensity."""

    def __init__(
        self,
        input_image: str = "input",
        output_image: str = "inverted",
        name: Optional[str] = None,
    ):
        """Initialize invert module.

        Parameters
        ----------
        input_image : str
            Name of input image in workspace.
        output_image : str
            Name to assign to output image.
        name : str, optional
            Module name.
        """
        super().__init__(name)
        self.input_image = input_image
        self.output_image = output_image

    @property
    def category(self) -> ModuleCategory:
        return ModuleCategory.IMAGE_PROCESSING

    def run(self, workspace: Workspace) -> None:
        image = workspace.get_image(self.input_image)

        if np.issubdtype(image.dtype, np.floating):
            inverted = 1.0 - image
        else:
            inverted = np.iinfo(image.dtype).max - image

        workspace.set_image(self.output_image, inverted)


class CropModule(Module):
    """Module that crops an image."""

    def __init__(
        self,
        input_image: str = "input",
        output_image: str = "cropped",
        top: int = 0,
        left: int = 0,
        height: int = 100,
        width: int = 100,
        name: Optional[str] = None,
    ):
        """Initialize crop module.

        Parameters
        ----------
        input_image : str
            Name of input image in workspace.
        output_image : str
            Name to assign to output image.
        top : int
            Top row coordinate.
        left : int
            Left column coordinate.
        height : int
            Height of crop region.
        width : int
            Width of crop region.
        name : str, optional
            Module name.
        """
        super().__init__(name)
        self.input_image = input_image
        self.output_image = output_image
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    @property
    def category(self) -> ModuleCategory:
        return ModuleCategory.IMAGE_PROCESSING

    def run(self, workspace: Workspace) -> None:
        image = workspace.get_image(self.input_image)
        cropped = image[
            self.top:self.top + self.height,
            self.left:self.left + self.width
        ].copy()
        workspace.set_image(self.output_image, cropped)
