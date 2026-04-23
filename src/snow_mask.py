"""Stage 4 lite: snow-region mask for seasonally-robust matching.

The cross-seasonal gap (winter drone footage vs summer/autumn
satellite tiles) makes feature matching unreliable on the snow itself
— large textureless regions where every matcher finds either no
features or random consensus matches. Stable across seasons:

  * roads        (asphalt or dirt, visible through snow)
  * tree clusters / forest edges
  * built-up structures
  * field boundaries (fences, hedges, watercourses)

The snow surface itself is unstable: heavily textureless under snow,
green / brown / yellow under summer satellite. Removing snow pixels
focuses the matcher on the seasonally-stable subset.

Detection is intentionally cheap and conservative — a single HSV
threshold rather than a learned segmenter:

    snow = (V > v_threshold) AND (S < s_threshold)

Defaults are tuned for GoPro footage at noon over snowy fields in
central Russia. Bright sun on snow reaches V > 0.9 with S < 0.10;
darker shadowed snow drops to V ~ 0.7 with S still < 0.15. The
threshold trades false positives (some bright sky/asphalt flagged as
snow) against false negatives (missed snow). Bias toward false
positives — losing a few matches is cheaper than including snow noise.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_snow_mask(
    img_bgr: np.ndarray,
    v_threshold: float = 0.80,
    s_threshold: float = 0.15,
    morph_kernel: int = 5,
) -> np.ndarray:
    """Return a uint8 0/255 mask where 255 = snow.

    Both thresholds are normalised to [0, 1]. ``morph_kernel`` size>0
    closes small holes inside snow regions and removes single-pixel
    speckle, producing cleaner mask boundaries; set to 0 to skip.
    """
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    snow = ((v > v_threshold) & (s < s_threshold)).astype(np.uint8) * 255
    if morph_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        snow = cv2.morphologyEx(snow, cv2.MORPH_CLOSE, kernel)
    return snow


def combine_with_aircraft_mask(
    snow_mask: np.ndarray, aircraft_mask: np.ndarray | None
) -> np.ndarray:
    """OR-combine snow + aircraft no-go regions into a single mask.

    Returns a uint8 0/255 mask where 255 marks pixels the matcher must
    NOT use. The matcher's aircraft_mask parameter consumes this format
    directly.
    """
    if aircraft_mask is None:
        return snow_mask
    if snow_mask.shape[:2] != aircraft_mask.shape[:2]:
        aircraft_mask = cv2.resize(
            aircraft_mask, (snow_mask.shape[1], snow_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return cv2.bitwise_or(snow_mask, aircraft_mask)
