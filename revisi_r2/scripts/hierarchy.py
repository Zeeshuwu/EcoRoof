"""Offline re-assembly of the SAM3 hierarchical pipeline from cached instances.

Replicates sam3_advanced.inference.segment_multiscale_advanced line by line
(level union, patch blending with adaptive consensus, spatial validation of
level 3 against level 1, early stop) and sam3_advanced.postprocess, but lets
the caller change the confidence threshold, the set of levels, and the
post-processing parameters without running SAM3 again.
"""
import json
import os

import numpy as np

from common import CACHE, REPO  # noqa: F401  (REPO puts the repo on sys.path)
from sam3_advanced.config import AdvancedConfig
from sam3_advanced.filtering import validate_with_context
from sam3_advanced.patching import blend_patches_weighted_average, generate_patches
from sam3_advanced.postprocess import post_process_mask

# PAPER_CONFIG (what actually ran for the manuscript): processor filter
# score > 0.5 (the AdvancedConfig 0.20/0.22/0.28 thresholds are dominated by
# it), min_road_size 50 px, road-shaped geometric filter ON, smoothing ON.
PAPER_PARAMS = {"tau": 0.5, "min_size": 50, "geo_filter": True}
COLLECT_THRESHOLD = 0.35  # processor threshold used while caching

# Calibration grid (7 x 3 x 2 = 42 combinations), centred on PAPER_PARAMS.
GRID_TAU = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
GRID_MIN_SIZE = [50, 150, 300]
GRID_GEO = [True, False]


def grid():
    return [{"tau": t, "min_size": m, "geo_filter": g}
            for t in GRID_TAU for m in GRID_MIN_SIZE for g in GRID_GEO]


def param_id(p):
    return "t%.2f_m%d_g%d" % (p["tau"], p["min_size"], int(p["geo_filter"]))


# ------------------------------------------------------------------ storage
def save_instances(path, instances):
    n = len(instances)
    level = np.array([r["level"] for r in instances], np.int8)
    bbox = np.array([r["bbox"] for r in instances], np.int32).reshape(n, 4)
    prompt = np.array([r["prompt_idx"] for r in instances], np.int16)
    score = np.array([r["score"] for r in instances], np.float32)
    y0 = np.array([r["y0"] for r in instances], np.int32)
    x0 = np.array([r["x0"] for r in instances], np.int32)
    h = np.array([r["crop"].shape[0] for r in instances], np.int32)
    w = np.array([r["crop"].shape[1] for r in instances], np.int32)
    packed = [np.packbits(r["crop"].ravel()) for r in instances]
    off = np.zeros(n + 1, np.int64)
    off[1:] = np.cumsum([len(p) for p in packed])
    bits = np.concatenate(packed) if packed else np.zeros(0, np.uint8)
    np.savez_compressed(path, level=level, bbox=bbox, prompt=prompt, score=score,
                        y0=y0, x0=x0, h=h, w=w, off=off, bits=bits)


class Instances:
    def __init__(self, path):
        z = np.load(path)
        for k in z.files:
            setattr(self, k, z[k])
        self.n = len(self.score)

    def crop(self, i):
        h, w = int(self.h[i]), int(self.w[i])
        b = np.unpackbits(self.bits[self.off[i]:self.off[i + 1]])[: h * w]
        return b.reshape(h, w).astype(bool)


# ------------------------------------------------------------------ assembly
def _union_level1(inst, shape, tau):
    out = np.zeros(shape, np.uint8)
    for i in np.flatnonzero((inst.level == 1) & (inst.score >= tau)):
        c = inst.crop(i)
        y0, x0 = int(inst.y0[i]), int(inst.x0[i])
        out[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] |= c
    return out


def _blend_level(inst, shape, tau, lvl, cfg):
    """Per-patch union, then blend_patches_weighted_average (same as pipeline).

    Only patches with at least one kept instance enter the blend, exactly like
    the pipeline (they are the only ones appended to patches_data)."""
    sel = np.flatnonzero((inst.level == lvl) & (inst.score >= tau))
    patches = {}
    for i in sel:
        key = tuple(int(v) for v in inst.bbox[i])
        x0p, y0p, x1p, y1p = key
        if key not in patches:
            patches[key] = np.zeros((y1p - y0p, x1p - x0p), np.float32)
        c = inst.crop(i)
        y0, x0 = int(inst.y0[i]), int(inst.x0[i])
        sub = patches[key][y0:y0 + c.shape[0], x0:x0 + c.shape[1]]
        np.maximum(sub, c.astype(np.float32), out=sub)
    # generate_patches() emits duplicate edge patches (e.g. width 1999, stride
    # 384 -> x=1536 and x=1920 both clamp to 1487..1999). The pipeline runs and
    # blends each copy, so a duplicated patch counts twice in the consensus.
    from collections import Counter
    size = cfg.patch_size_level2 if lvl == 2 else cfg.patch_size_level3
    mult = Counter(tuple(int(v) for v in pi["bbox"])
                   for pi in generate_patches(shape, size, cfg.overlap_ratio))
    data = [{"mask": m, "bbox": k} for k, m in patches.items() for _ in range(mult.get(k, 1))]
    merged, _ = blend_patches_weighted_average(
        data, shape, cfg.consensus_threshold_multi, cfg.consensus_threshold_single)
    return merged


def _quiet(fn, *a, **k):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def level_masks(inst, shape, tau):
    """Raw per-level masks at threshold tau (before cross-level merging)."""
    cfg = AdvancedConfig()
    l1 = _union_level1(inst, shape, tau)
    l2 = _blend_level(inst, shape, tau, 2, cfg)
    l3 = _blend_level(inst, shape, tau, 3, cfg)
    return l1, l2, l3


def merge_levels(l1, l2, l3, use=(1, 2, 3), l3_skipped=False):
    """Cross-level merge exactly as the pipeline, for any subset of levels.

    Pipeline rules reproduced: level 2 is not validated (validate_level2=False);
    level 3 is validated against level 1 (validate_level3=True) - when level 1
    is not part of the ablation there is no context mask, so level 3 enters
    unvalidated; early stop skips level 3 when L1+L2 coverage >= 95 %."""
    cfg = AdvancedConfig()
    shape = l1.shape
    zero = np.zeros(shape, np.uint8)
    a = l1 if 1 in use else zero
    b = l2 if 2 in use else zero
    final = np.maximum(a, b)
    if 3 in use and not l3_skipped:
        cov = final.mean()
        if not (1 in use and 2 in use and cov >= cfg.early_stop_coverage["level2"]):
            if 1 in use and cfg.enable_spatial_validation and cfg.validate_level3:
                c = _quiet(validate_with_context, l3, l1, min_overlap=cfg.min_overlap_with_level1)
            else:
                c = l3
            final = np.maximum(final, c)
    return final.astype(np.uint8)


def postprocess(mask, min_size=50, geo_filter=True):
    cfg = AdvancedConfig()
    cfg.min_road_size = int(min_size)
    cfg.enable_geometric_filter = bool(geo_filter)
    return _quiet(post_process_mask, mask, cfg, verbose=False)


def load_meta(key):
    with open(os.path.join(CACHE, "%s_meta.json" % key), encoding="utf-8") as f:
        return json.load(f)
