# Tools

GAZE ships 25 built-in tools: 23 visual tools that operate on the active image
and 2 search tools that retrieve external evidence. Visual tools come from
`create_visual_tools()` (`src/gaze/tools/visual.py`); search tools from
`create_search_tools()` (`src/gaze/tools/search.py`).

Visual tools are enabled with `use_tools=True` and search tools with
`use_web_search=True`. Tools are only offered in multi-turn mode
(`max_turns > 1`); the final turn always withholds them. Any tool can be
excluded by name via the `disabled_tools` constructor argument.

## Coordinate and intensity effects

GAZE classifies tools by whether they change how subsequent
measurements relate to the original image (`base.py`):

- **Coordinate-modifying** (`_COORD_MODIFYING_TOOLS`): `crop`, `zoom`,
  `rotate`, `flip_horizontal`, `flip_vertical`. After any of these, bounding
  boxes no longer correspond to original pixel coordinates.
- **Intensity-modifying** (`_INTENSITY_MODIFYING_TOOLS`): `threshold`,
  `window_level`, `equalize_histogram`, `adaptive_equalize`, `invert`,
  `detect_edges`, `symmetry_diff`, `morphological`, `denoise`,
  `adjust_contrast`, `adjust_brightness`, `adjust_sharpness`. After any of
  these, pixel values no longer represent original tissue intensities, so
  `get_intensity_stats` and `intensity_profile` reflect the transformed data.

On the final turn GAZE re-attaches the original image and warns the
model when either class of tool was used. A successful `reset` clears both
flags. Tools marked **read-only** below render an image (or return numbers)
without mutating the registry's image state, so they do not set either flag.

The numeric ranges quoted below are the defaults from `ImageProcessingConfig`
(`src/gaze/config.py`). They are configurable: overriding the config changes
both the validation bounds and the ranges advertised to the model.

## Visual tools

### Geometry

| Tool | Purpose | Parameters | Effect |
|------|---------|------------|--------|
| `zoom` | Magnify the image for detail analysis | `factor` (number, 0.5-4.0) | coordinate |
| `crop` | Extract a rectangular region | `box` ([x1,y1,x2,y2], normalized 0-1; pixel coords auto-converted) | coordinate |
| `rotate` | Rotate the image 90 degrees | `clockwise` (boolean, default `true`) | coordinate |
| `flip_horizontal` | Mirror left-right (reverses laterality) | none | coordinate |
| `flip_vertical` | Mirror top-bottom (reverses superior/inferior) | none | coordinate |

### Intensity and contrast

| Tool | Purpose | Parameters | Effect |
|------|---------|------------|--------|
| `adjust_contrast` | Enhance or reduce contrast | `factor` (number, 0.5-3.0; 1.0 = no change) | intensity |
| `adjust_brightness` | Adjust brightness | `factor` (number, 0.5-2.0; 1.0 = no change) | intensity |
| `adjust_sharpness` | Adjust sharpness | `factor` (number, 0.1-3.0; 1.0 = original) | intensity |
| `threshold` | Intensity windowing, rescaled to 0-255 (grayscale) | `lower` (integer, 0-254), `upper` (integer, 1-255, must exceed `lower`; window width >= 50) | intensity |
| `window_level` | Clinical window/level (grayscale) | `preset` (enum) OR both `center` (integer) and `width` (integer, >= 50) | intensity |
| `equalize_histogram` | Global histogram equalization (grayscale) | none | intensity |
| `adaptive_equalize` | CLAHE local contrast (grayscale) | `clip_limit` (number, 1.0-4.0, default 2.0), `tile_size` (integer, 2-32, default 8) | intensity |
| `invert` | Invert intensities, negative (grayscale) | none | intensity |
| `denoise` | Gaussian blur for noise reduction | `sigma` (number, 0.5-5.0) | intensity |

`window_level` requires either a `preset` or both `center` and `width`; if a
preset is supplied it overrides `center`/`width`. The available presets are
the keys of `WINDOW_PRESETS` in `visual.py`. MRI presets (for 8-bit pixel
values) are `brain`/`mri_brain`, `flair`/`mri_flair`, `t2`/`mri_t2`,
`stroke`, `posterior_fossa`. CT presets (Hounsfield units) are `ct_brain`,
`ct_subdural`, `ct_bone`, `ct_soft_tissue`, `ct_stroke`,
`ct_posterior_fossa`. Applying CT presets to 8-bit MRI data is rejected when
it would compress the dynamic range below the minimum window width.

### Edges and morphology

| Tool | Purpose | Parameters | Effect |
|------|---------|------------|--------|
| `detect_edges` | Edge map via Sobel or Laplacian (grayscale) | `method` (enum: `sobel`, `laplacian`; default `sobel`) | intensity |
| `symmetry_diff` | Left-right symmetry difference map (grayscale) | none | intensity |
| `morphological` | Erode/dilate/open/close (grayscale) | `operation` (enum: `erode`, `dilate`, `open`, `close`), `iterations` (integer, 1-5, default 1), `threshold_value` (integer, 0-255, optional binarization) | intensity |

### Measurement (read-only)

| Tool | Purpose | Parameters | Effect |
|------|---------|------------|--------|
| `get_intensity_stats` | Mean/std/min/max/median + 10-bin histogram | `box` ([x1,y1,x2,y2], normalized 0-1, optional; full image if omitted) | read-only |
| `intensity_profile` | Sample intensities along a line | `point1`, `point2` ([x,y], normalized 0-1) | read-only |
| `measure` | Euclidean distance between two points (pixels) | `point1`, `point2` ([x,y], normalized 0-1) | read-only |

These tools report against the current image. When the image has been
modified they append a note recommending `reset` for original-space values.

### Annotation and state (read-only)

| Tool | Purpose | Parameters | Effect |
|------|---------|------------|--------|
| `show_grid` | Overlay a labeled reference grid (e.g. A1, B2) | `divisions` (integer, 2-8) | read-only |
| `annotate_region` | Draw a bounding box overlay for propose-and-verify | `box` ([x1,y1,x2,y2], normalized 0-1), `color` (enum: red, green, yellow, blue, white, cyan, magenta; default red), `label` (string, optional) | read-only |
| `reset` | Restore the original image, discarding all modifications | none | clears both flags |

`show_grid` and `annotate_region` render an image for the model to see
without changing the registry's image state. `reset` restores the original
image and clears the coordinate- and intensity-modified flags.

## Search tools

Enabled with `use_web_search=True`. These do not require an image and can run
in text-only sessions. Results are length-limited and sanitized before being
returned to the model.

| Tool | Purpose | Parameters |
|------|---------|------------|
| `search_web` | Search PubMed for medical literature and guidelines | `query` (string), `search_type` (enum: `diagnosis`, `research`, `guidelines`, `anatomy`, `treatment`, `differential`, `general`; default `general`) |
| `search_images` | Search NIH Open-i for reference medical images | `query` (string), `modality` (enum: MRI, CT, X-ray, Ultrasound, PET, Mammography, any; default `any`), `body_part` (enum: brain, head, chest, abdomen, spine, pelvis, cardiac, any; default `any`) |

Search timeouts, retry counts, rate limits, and endpoint URLs are governed by
`SearchConfig` (see [Configuration](configuration.md)). The PubMed and Open-i
base URLs are validated against an allowlist of NCBI/NLM hostnames.
