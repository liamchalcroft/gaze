# Exceptions

The `GazeError` hierarchy raised across the library. Catch specific subtypes
rather than the base class so that distinct failures (model errors, schema
validation, tool failures) can be handled separately.

::: gaze.exceptions
