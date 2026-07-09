"""
Window partitioning for NaDiT's windowed 3D attention, ported from
dit_3b/window.py and dit_7b/window.py (byte-identical between variants).

Both variants target a fixed window *count* per axis (``window_counts``,
typically (4, 3, 3) for (t, h, w)) rather than a fixed pixel size. Because
the token grid is native-resolution (not necessarily 720p), the window pixel
size is first computed as if the video were rescaled to 45x80 (720p latent
grid at patch=2 -> 720/16=45, 1280/16=80... the exact constant is inherited
verbatim from the source, see ``_rescale_window_size``), then the *actual*
window/​grid counts at the real resolution are re-derived from that pixel
size via ceil-division. This means edge windows along any axis can end up
smaller than the interior windows -- callers must group windows by shape
before batching them into a single attention call.
"""

import math
from math import ceil
from typing import Dict, List, Tuple

Slice3 = Tuple[slice, slice, slice]


def _window_pixel_size(h: int, w: int, wt_count: int, wh_count: int, ww_count: int, t: int) -> Tuple[int, int, int]:
    # sqrt(45 * 80 / (h * w)) rescales (h, w) to a 720p-equivalent grid before
    # dividing into `wh_count` x `ww_count` windows -- this keeps the window
    # *pixel* footprint roughly constant across differing input resolutions.
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / wh_count), ceil(resized_w / ww_count)
    wt = ceil(min(t, 30) / wt_count)
    return wt, wh, ww


def compute_windows(
    t: int,
    h: int,
    w: int,
    window_counts: Tuple[int, int, int] = (4, 3, 3),
    shifted: bool = False,
) -> List[Slice3]:
    """
    Returns a flat list of (t_slice, h_slice, w_slice) windows that exactly
    tile the (t, h, w) volume (no overlap, full coverage), in the same
    iteration order as the original ``make_720Pwindows_bysize`` /
    ``make_shifted_720Pwindows_bysize``.
    """
    wt_count, wh_count, ww_count = window_counts
    wt, wh, ww = _window_pixel_size(h, w, wt_count, wh_count, ww_count, t)

    if not shifted:
        nt, nh, nw = ceil(t / wt), ceil(h / wh), ceil(w / ww)
        return [
            (
                slice(it * wt, min((it + 1) * wt, t)),
                slice(ih * wh, min((ih + 1) * wh, h)),
                slice(iw * ww, min((iw + 1) * ww, w)),
            )
            for iw in range(nw)
            if min((iw + 1) * ww, w) > iw * ww
            for ih in range(nh)
            if min((ih + 1) * wh, h) > ih * wh
            for it in range(nt)
            if min((it + 1) * wt, t) > it * wt
        ]

    st, sh, sw = (
        0.5 if wt < t else 0,
        0.5 if wh < h else 0,
        0.5 if ww < w else 0,
    )
    nt, nh, nw = ceil((t - st) / wt), ceil((h - sh) / wh), ceil((w - sw) / ww)
    nt, nh, nw = (
        nt + 1 if st > 0 else 1,
        nh + 1 if sh > 0 else 1,
        nw + 1 if sw > 0 else 1,
    )
    return [
        (
            slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)),
            slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)),
            slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)),
        )
        for iw in range(nw)
        if min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
        for ih in range(nh)
        if min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
        for it in range(nt)
        if min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)
    ]


def group_windows(windows: List[Slice3]) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Groups window indices by their (tw, hw, ww) token-count shape, so that
    same-shape windows can be stacked into one batch and processed with a
    single dense ``optimized_attention`` call.
    """
    groups: Dict[Tuple[int, int, int], List[int]] = {}
    for i, (ts, hs, ws) in enumerate(windows):
        shape = (ts.stop - ts.start, hs.stop - hs.start, ws.stop - ws.start)
        groups.setdefault(shape, []).append(i)
    return groups
