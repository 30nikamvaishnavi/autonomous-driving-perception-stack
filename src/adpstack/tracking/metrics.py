"""MOT metrics: MOTA, IDF1 and ID switches (self-contained, IoU-based matching, no external deps).

For official benchmark numbers use TrackEval/py-motmetrics on BDD100K MOT; this module gives the same
quantities for quick ablations and unit tests.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .bytetrack_wrapper import iou_matrix

# frames: list over time of (ids (N,), xyxy (N,4))
Frames = list[tuple[np.ndarray, np.ndarray]]


def mot_metrics(gt: Frames, pred: Frames, iou_thr: float = 0.5) -> dict[str, float]:
    """Compute MOTA, IDF1, IDP, IDR, ID switches, FP, FN over aligned GT / prediction sequences."""
    fp = fn = idsw = n_gt = tp = 0
    last_match: dict[int, int] = {}
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    gt_len: dict[int, int] = defaultdict(int)
    pr_len: dict[int, int] = defaultdict(int)
    for (gid, gb), (pid, pb) in zip(gt, pred, strict=True):
        n_gt += len(gid)
        for g in gid:
            gt_len[int(g)] += 1
        for p in pid:
            pr_len[int(p)] += 1
        if len(gid) == 0 or len(pid) == 0:
            fn += len(gid)
            fp += len(pid)
            continue
        iou = iou_matrix(np.asarray(gb, float), np.asarray(pb, float))
        cost = np.where(iou >= iou_thr, 1.0 - iou, 1e6)
        r, c = linear_sum_assignment(cost)
        matched_g, matched_p = set(), set()
        for i, j in zip(r, c, strict=True):
            if cost[i, j] >= 1e6:
                continue
            g, p = int(gid[i]), int(pid[j])
            matched_g.add(i)
            matched_p.add(j)
            tp += 1
            pair_counts[(g, p)] += 1
            if g in last_match and last_match[g] != p:
                idsw += 1
            last_match[g] = p
        fn += len(gid) - len(matched_g)
        fp += len(pid) - len(matched_p)
    mota = 1.0 - (fn + fp + idsw) / max(n_gt, 1)
    # IDF1 via global optimal ID assignment
    gids, pids = sorted(gt_len), sorted(pr_len)
    idtp = 0
    if gids and pids:
        M = np.zeros((len(gids), len(pids)))
        gi, pi = {g: i for i, g in enumerate(gids)}, {p: i for i, p in enumerate(pids)}
        for (g, p), c in pair_counts.items():
            M[gi[g], pi[p]] = c
        r, c = linear_sum_assignment(-M)
        idtp = int(M[r, c].sum())
    n_pr = sum(pr_len.values())
    idp = idtp / max(n_pr, 1)
    idr = idtp / max(n_gt, 1)
    idf1 = 2 * idtp / max(n_gt + n_pr, 1)
    return {"MOTA": mota, "IDF1": idf1, "IDP": idp, "IDR": idr, "ID_switches": idsw,
            "FP": fp, "FN": fn, "TP": tp, "GT": n_gt}
