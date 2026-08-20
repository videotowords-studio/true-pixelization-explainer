class PixelizerError(Exception):
    """Base error for expected pixelizer failures."""


class ConfigurationError(PixelizerError):
    """The requested output contract is invalid."""


class InputImageError(PixelizerError):
    """The source image or mask cannot be decoded safely."""


class ProcessingError(PixelizerError):
    """The deterministic processing pipeline could not finish."""


class QualityError(PixelizerError):
    """The exported result failed a hard quality check."""


class ExportError(PixelizerError):
    """The validated artifacts could not be written safely."""
