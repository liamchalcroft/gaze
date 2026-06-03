# Configuration

GAZE configuration lives in `src/gaze/config.py`. All configuration objects
are frozen dataclasses: they validate their fields in `__post_init__` and
cannot be mutated after construction. To change configuration you build a new
object and either install it as the process default or apply it for the
duration of a block.

The configuration tree has one root and three sub-configs:

```python
from gaze import GazeConfig

config = GazeConfig()        # all defaults
config.image                 # ImageProcessingConfig
config.cache                 # CacheConfig
config.search                # SearchConfig
```

## GazeConfig

The root object. It holds the three sub-configs and is what `get_config()`
returns. Build a customized tree by passing the sub-configs you want to
override; the rest fall back to defaults.

```python
from gaze import GazeConfig, CacheConfig, SearchConfig

config = GazeConfig(
    cache=CacheConfig(max_cache_size=1000),
    search=SearchConfig(timeout_seconds=60),
)

print(config.image.max_zoom_factor)        # 4.0 (default, untouched)
print(config.cache.cache_duration_seconds) # 300
```

## ImageProcessingConfig

Bounds for the visual tools and image handling. The validation ranges here
are the same numbers advertised to the model in tool documentation (see
[Tools](tools.md)), so changing them changes both enforcement and the prompt.

Selected fields and their defaults:

| Field | Default | Meaning |
|-------|---------|---------|
| `min_image_size` | 10 | Minimum dimension for images and crops (px) |
| `max_image_dimension` | 16384 | Maximum allowed width or height (px) |
| `min_zoom_factor` / `max_zoom_factor` | 0.5 / 4.0 | Allowed `zoom` factor range |
| `min_contrast_factor` / `max_contrast_factor` | 0.5 / 3.0 | Allowed `adjust_contrast` range |
| `min_brightness_factor` / `max_brightness_factor` | 0.5 / 2.0 | Allowed `adjust_brightness` range |
| `min_sharpness_factor` / `max_sharpness_factor` | 0.1 / 3.0 | Allowed `adjust_sharpness` range |
| `min_threshold_window` | 50 | Minimum `threshold` window width |
| `min_window_width` | 50 | Minimum `window_level` width |
| `min_gaussian_sigma` / `max_gaussian_sigma` | 0.5 / 5.0 | Allowed `denoise` sigma range |
| `min_clahe_clip_limit` / `max_clahe_clip_limit` | 1.0 / 4.0 | Allowed `adaptive_equalize` clip limit |
| `max_clahe_tile_size` | 32 | Maximum CLAHE tile grid size |
| `max_morphological_iterations` | 5 | Maximum `morphological` iterations |
| `max_grid_divisions` | 8 | Maximum `show_grid` divisions per axis |
| `default_jpeg_quality` | 85 | JPEG quality for encoding (1-100) |

Constructing an out-of-range config raises `ValueError` immediately. For
example, `ImageProcessingConfig(min_zoom_factor=5.0, max_zoom_factor=4.0)`
fails because the minimum must be below the maximum.

## CacheConfig

Controls the in-memory TTL cache.

| Field | Default | Meaning |
|-------|---------|---------|
| `max_cache_size` | 500 | Maximum number of cached entries |
| `cache_duration_seconds` | 300 | Time-to-live per entry (seconds) |
| `evict_ratio` | 0.5 | Fraction evicted when over the limit (0.0-1.0 exclusive) |

## SearchConfig

Parameters for PubMed and Open-i retrieval.

| Field | Default | Meaning |
|-------|---------|---------|
| `timeout_seconds` | 30 | Request timeout |
| `max_retries` | 3 | Maximum retry attempts |
| `rate_limit_delay_seconds` | 1.0 | Delay between API calls |
| `max_content_preview_length` | 500 | Characters per result preview |
| `max_snippet_length` | 100 | Characters per extracted snippet |
| `max_content_for_llm` | 5000 | Total characters of formatted results |
| `ncbi_base_url` | NCBI E-utilities URL | PubMed endpoint |
| `openi_base_url` | Open-i search URL | Image search endpoint |

Both base URLs are validated for SSRF protection: they must use HTTPS and
their hostnames must appear in the built-in NCBI/NLM allowlist, which blocks
DNS-rebinding to internal services.

## Applying configuration

There are two ways to install a configuration, exported from the top-level
package.

### Process-wide default

`set_config()` replaces the global default for every thread and task that
does not have an active override. `get_config()` reads the effective config,
and `reset_config()` restores a fresh `GazeConfig()` (useful in test
teardown).

```python
from gaze import set_config, get_config, reset_config, GazeConfig, CacheConfig

set_config(GazeConfig(cache=CacheConfig(max_cache_size=1000)))
assert get_config().cache.max_cache_size == 1000

reset_config()
assert get_config().cache.max_cache_size == 500
```

### Task-scoped override

`config_context()` is a context manager backed by a `ContextVar`. Inside the
block, `get_config()` returns the temporary config; on exit the previous
value is restored. Because the override lives in a `ContextVar`, concurrent
asyncio tasks and threads each see their own override and never clobber the
process-wide default. Nested contexts restore correctly.

```python
import asyncio
from gaze import config_context, get_config, GazeConfig, ImageProcessingConfig

async def restricted_zoom(image, metadata, processor):
    # Tighten the zoom bound for this task only.
    tight = GazeConfig(image=ImageProcessingConfig(max_zoom_factor=2.0))
    with config_context(tight):
        # get_config().image.max_zoom_factor == 2.0 here, and the zoom tool
        # rejects factors above 2.0 for the duration of this analyze() call.
        return await processor.analyze(images=image, metadata=metadata)

async def main(image, metadata, processor):
    # Two tasks with independent overrides running concurrently.
    relaxed = GazeConfig(image=ImageProcessingConfig(max_zoom_factor=4.0))
    async def relaxed_zoom():
        with config_context(relaxed):
            return await processor.analyze(images=image, metadata=metadata)

    await asyncio.gather(
        restricted_zoom(image, metadata, processor),
        relaxed_zoom(),
    )
    # Outside both blocks, get_config() is back to the process default.
    assert get_config().image.max_zoom_factor == 4.0
```

Prefer `config_context()` for per-request or per-task tuning and reserve
`set_config()` for genuinely global, program-wide changes. Do not attempt to
mutate a config object in place; the frozen dataclasses forbid it.
