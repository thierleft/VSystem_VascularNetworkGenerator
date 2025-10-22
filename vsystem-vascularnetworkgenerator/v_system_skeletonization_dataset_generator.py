#!/usr/bin/env python3
"""
V-System–style vascular dataset generator for deep learning skeletonization / centreline prediction.

Per-sample outputs live in a single folder with metadata in the name, e.g.:
  sample_0003__shape-1024x1024x1024__fmt-nifti__pix-1x1x1__curved-1__comp-0p6__seed-42

Each folder contains:
- Volume(s):
  - Zarr:   volume.zarr/ (vessels, centreline)
  - NIfTI:  vessels.nii.gz, centreline.nii.gz
  - TIFF:   tiff/vessels/z00000.tif, tiff/centreline/z00000.tif
- CSVs: nodes.csv, edges.csv, edge_samples.csv, branches.csv, adjacency.csv

Key options
- Inward warm-up:        --root-inward, --first-seg-len
- Root control:          --initial-diam, --min-diam-stop
- Loose avoidance:       --surface-clearance (voxels) + --clearance-factor
- Robust tip growth:     --skip-t-head
- Forced branching:      --force-branch-depth  (guarantee branching for first N generations)
"""

from __future__ import annotations
import argparse
import csv
import math
import os
import gc
import shutil
import time
from dataclasses import dataclass, asdict
import concurrent.futures as cf
from typing import List, Tuple, Dict, Optional

import numpy as np

# Zarr (optional)
try:
    import zarr  # type: ignore
    from numcodecs import Blosc  # type: ignore
except Exception:
    zarr = None

# Optional formats: NIfTI and TIFF slices
try:
    import nibabel as nib  # type: ignore
except Exception:
    nib = None

try:
    import tifffile  # type: ignore
except Exception:
    tifffile = None

# TQDM (optional)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# Optional Numba for speed
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def wrapper(f):
            return f
        return wrapper

# --------------------------- Geometry helpers ---------------------------- #

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    # Marsaglia method
    u = rng.uniform(-1, 1)
    th = rng.uniform(0, 2 * math.pi)
    s = math.sqrt(1 - u * u)
    return np.array([s * math.cos(th), s * math.sin(th), u], dtype=np.float32)

def orthonormal_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = unit(n)
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v = unit(np.cross(n, a))
    u = unit(np.cross(v, n))
    return u, v, n

def rotate_towards(dir_vec: np.ndarray, max_angle_rad: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate dir_vec by a random angle up to max_angle_rad around a random axis."""
    axis_guess = random_unit_vector(rng)
    axis = unit(np.cross(axis_guess, dir_vec))
    if np.linalg.norm(axis) < 1e-6:
        cand = np.cross(dir_vec, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(cand) < 1e-6:
            cand = np.cross(dir_vec, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis = unit(cand)
    angle = rng.uniform(-max_angle_rad, max_angle_rad)
    return rotate_vector(dir_vec, axis, angle)

def rotate_vector(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    # Rodrigues' rotation formula
    axis = unit(axis)
    v_par = np.dot(v, axis) * axis
    v_perp = v - v_par
    w = np.cross(axis, v_perp)
    return v_par + v_perp * math.cos(angle) + w * math.sin(angle)

# --------------------------- Graph primitives --------------------------- #

@dataclass
class Node:
    id: int
    xyz: Tuple[float, float, float]  # (x, y, z)
    radius: float                    # radius (voxels)

@dataclass
class Edge:
    id: int
    start: int
    end: int
    samples: List[Tuple[float, float, float]]  # centreline samples (x,y,z)
    radii: List[float]                          # radius per sample

class VascularGraph:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Edge] = {}
        self.adj: Dict[int, List[int]] = {}
        self._next_node_id = 0
        self._next_edge_id = 0
        self.roots: List[int] = []  # seed nodes

    def add_node(self, xyz: np.ndarray, radius: float) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        n = Node(id=nid, xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])), radius=float(radius))
        self.nodes[nid] = n
        self.adj.setdefault(nid, [])
        return nid

    def add_edge(self, start: int, end: int, samples: List[Tuple[float, float, float]], radii: List[float]) -> int:
        eid = self._next_edge_id
        self._next_edge_id += 1
        e = Edge(id=eid, start=start, end=end, samples=samples, radii=radii)
        self.edges[eid] = e
        self.adj[start].append(end)
        self.adj[end].append(start)
        return eid

    def branch_endpoints(self) -> List[Tuple[int, int, List[int]]]:
        """Return (start_node_id, end_node_id, path_node_ids) for maximal paths whose internal nodes have degree 2."""
        degree = {nid: len(neigh) for nid, neigh in self.adj.items()}
        endpoints = [nid for nid, d in degree.items() if d != 2]
        visited_edges = set()
        branches = []
        node_to_edges: Dict[int, List[int]] = {nid: [] for nid in self.nodes}
        for eid, e in self.edges.items():
            node_to_edges[e.start].append(eid)
            node_to_edges[e.end].append(eid)
        for s in endpoints:
            for eid in node_to_edges.get(s, []):
                if eid in visited_edges:
                    continue
                path_nodes = [s]
                curr_eid = eid
                curr_node = s
                while True:
                    visited_edges.add(curr_eid)
                    e = self.edges[curr_eid]
                    nxt = e.end if curr_node == e.start else e.start
                    path_nodes.append(nxt)
                    if len(self.adj[nxt]) != 2:
                        branches.append((path_nodes[0], nxt, path_nodes))
                        break
                    incident = node_to_edges[nxt]
                    next_eid = incident[0] if incident[1] == curr_eid else incident[1]
                    curr_node = nxt
                    curr_eid = next_eid
        return branches

# ---------------------------- Curve builders ---------------------------- #

def quadratic_bezier(p0: np.ndarray, p1: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a = np.outer((1 - t) ** 2, p0)
    b = np.outer(2 * (1 - t) * t, c)
    d = np.outer(t ** 2, p1)
    return a + b + d

def catmull_rom_spline(points: np.ndarray, samples_per_seg: int) -> np.ndarray:
    """Catmull-Rom spline through points [m,3] → sampled points."""
    if len(points) < 2:
        return points.copy()
    pts = np.vstack([points[0], points, points[-1]])  # clamp endpoints
    out = []
    for i in range(pts.shape[0] - 3):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        for j in range(samples_per_seg):
            t = j / float(samples_per_seg)
            t2, t3 = t * t, t * t * t
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(q)
    out.append(points[-1])
    return np.asarray(out, dtype=np.float32)

# ---------------------- Distance & rasterization ------------------------ #

@njit
def _distance_point_to_segment(px, py, pz, ax, ay, az, bx, by, bz):
    abx = bx - ax; aby = by - ay; abz = bz - az
    apx = px - ax; apy = py - ay; apz = pz - az
    ab2 = abx*abx + aby*aby + abz*abz
    if ab2 == 0.0:
        dx = px - ax; dy = py - ay; dz = pz - az
        return math.sqrt(dx*dx + dy*dy + dz*dz), 0.0
    t = (apx*abx + apy*aby + apz*abz) / ab2
    if t < 0.0: t = 0.0
    elif t > 1.0: t = 1.0
    cx = ax + t * abx; cy = ay + t * aby; cz = az + t * abz
    dx = px - cx; dy = py - cy; dz = pz - cz
    return math.sqrt(dx*dx + dy*dy + dz*dz), t

class TileView(np.ndarray):
    """Attach absolute origin to array views for painting."""
    def __new__(cls, arr: np.ndarray, origin: Tuple[int, int, int]):
        obj = np.asarray(arr).view(cls)
        obj._absolute_origin = origin  # type: ignore[attr-defined]
        return obj

def paint_frustum_region(vol: np.ndarray,
                         cl_vol: Optional[np.ndarray],
                         p0: np.ndarray,
                         p1: np.ndarray,
                         r0: float,
                         r1: float):
    """
    Paint a truncated cone (frustum) for segment p0->p1 with radii r0, r1 into a dense subvolume view.
    vol/cl_vol are (Z,Y,X); p0/p1 are (X,Y,Z).
    """
    z0_abs, y0_abs, x0_abs = vol._absolute_origin  # type: ignore[attr-defined]
    z_dim, y_dim, x_dim = vol.shape
    zz, yy, xx = np.meshgrid(
        np.arange(z0_abs, z0_abs + z_dim, dtype=np.float32),
        np.arange(y0_abs, y0_abs + y_dim, dtype=np.float32),
        np.arange(x0_abs, x0_abs + x_dim, dtype=np.float32),
        indexing='ij'
    )
    ax, ay, az = p0[0], p0[1], p0[2]
    bx, by, bz = p1[0], p1[1], p1[2]
    ab = np.array([bx-ax, by-ay, bz-az], dtype=np.float32)
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-8:  # degenerate
        rr = r0
        dist = np.sqrt((xx-ax)**2 + (yy-ay)**2 + (zz-az)**2)
        mask = dist <= rr + 0.5
        vol[mask] = 1
        if cl_vol is not None:
            cx, cy, cz = int(round(ax)), int(round(ay)), int(round(az))
            if (z0_abs <= cz < z0_abs+z_dim) and (y0_abs <= cy < y0_abs+y_dim) and (x0_abs <= cx < x0_abs+x_dim):
                cl_vol[cz - z0_abs, cy - y0_abs, cx - x0_abs] = 1
        return

    apx = xx - ax; apy = yy - ay; apz = zz - az
    t = (apx*(bx-ax) + apy*(by-ay) + apz*(bz-az)) / ab2
    t = np.clip(t, 0.0, 1.0)
    cx = ax + t * (bx - ax); cy = ay + t * (by - ay); cz = az + t * (bz - az)
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2)
    rr = r0 + t * (r1 - r0)
    mask = dist <= rr + 0.5
    vol[mask] = 1

    if cl_vol is not None:
        steps = max(int(math.ceil(np.linalg.norm([bx-ax, by-ay, bz-az])) * 2), 1)
        for i in range(steps + 1):
            tt = i / float(steps)
            cx_f = ax + tt * (bx - ax)
            cy_f = ay + tt * (by - ay)
            cz_f = az + tt * (bz - az)
            cx_i, cy_i, cz_i = int(round(cx_f)), int(round(cy_f)), int(round(cz_f))
            if (z0_abs <= cz_i < z0_abs + z_dim and
                y0_abs <= cy_i < y0_abs + y_dim and
                x0_abs <= cx_i < x0_abs + x_dim):
                cl_vol[cz_i - z0_abs, cy_i - y0_abs, cx_i - x0_abs] = 1

def paint_edge_to_zarr(zarr_vol, zarr_cl, edge: Edge, chunk: Tuple[int, int, int], shape: Tuple[int, int, int]):
    """Paint one edge into array storage (works for Zarr arrays or numpy.memmap)."""
    pts = np.asarray(edge.samples, dtype=np.float32)
    radii = np.asarray(edge.radii, dtype=np.float32)
    for i in range(len(pts) - 1):
        p0 = pts[i]; p1 = pts[i + 1]
        r0 = float(radii[i]); r1 = float(radii[i + 1])
        minc = np.floor(np.minimum(p0, p1) - max(r0, r1) - 1).astype(int)
        maxc = np.ceil(np.maximum(p0, p1) + max(r0, r1) + 1).astype(int)
        minc = np.maximum(minc, 0)
        maxc = np.minimum(maxc, np.array(shape) - 1)
        zmin, ymin, xmin = minc[2], minc[1], minc[0]
        zmax, ymax, xmax = maxc[2], maxc[1], maxc[0]
        for z0 in range(zmin, zmax + 1, chunk[0]):
            for y0 in range(ymin, ymax + 1, chunk[1]):
                for x0 in range(xmin, xmax + 1, chunk[2]):
                    z1 = min(z0 + chunk[0], zmax + 1)
                    y1 = min(y0 + chunk[1], ymax + 1)
                    x1 = min(x0 + chunk[2], xmax + 1)
                    sub = zarr_vol[z0:z1, y0:y1, x0:x1][:]
                    sub_view = TileView(sub, origin=(z0, y0, x0))
                    if zarr_cl is not None:
                        sub_cl = zarr_cl[z0:z1, y0:y1, x0:x1][:]
                        sub_cl_view = TileView(sub_cl, origin=(z0, y0, x0))
                    else:
                        sub_cl_view = None
                    paint_frustum_region(sub_view, sub_cl_view,
                                         p0=np.array([p0[0], p0[1], p0[2]], dtype=np.float32),
                                         p1=np.array([p1[0], p1[1], p1[2]], dtype=np.float32),
                                         r0=r0, r1=r1)
                    zarr_vol[z0:z1, y0:y1, x0:x1] = sub_view
                    if zarr_cl is not None:
                        zarr_cl[z0:z1, y0:y1, x0:x1] = sub_cl_view

# ---------------------- Spatial self-avoidance -------------------------- #

class SpatialHash3D:
    def __init__(self, cell_size: float = 16.0):
        self.cell = float(cell_size)
        self.grid: Dict[Tuple[int,int,int], List[Tuple[float,float,float,float]]] = {}

    def _key(self, x: float, y: float, z: float) -> Tuple[int,int,int]:
        return (int(math.floor(x / self.cell)),
                int(math.floor(y / self.cell)),
                int(math.floor(z / self.cell)))

    def insert(self, x: float, y: float, z: float, r: float):
        k = self._key(x,y,z)
        self.grid.setdefault(k, []).append((x,y,z,r))

    def insert_polyline(self, pts: np.ndarray, radii: np.ndarray):
        for (x,y,z), r in zip(pts, radii):
            self.insert(float(x), float(y), float(z), float(r))

    def nearby(self, x: float, y: float, z: float, radius: float):
        reach = int(math.ceil(radius / self.cell))
        kx, ky, kz = self._key(x,y,z)
        for dz in range(-reach, reach+1):
            for dy in range(-reach, reach+1):
                for dx in range(-reach, reach+1):
                    for rec in self.grid.get((kx+dx, ky+dy, kz+dz), []):
                        yield rec

def _clearance_threshold(r: float, nr: float, *,
                         mode: str, cf: float,
                         cap: Optional[float],
                         surface: float) -> float:
    if mode == 'min':
        base = min(r, nr) * cf
    elif mode == 'geom':
        base = math.sqrt(max(r, 0.0) * max(nr, 0.0)) * cf
    else:  # 'sum'
        base = (r + nr) * cf
    if cap is not None:
        base = min(base, cap)
    return base + surface

def segment_is_clear(p0: np.ndarray, p1: np.ndarray, r0: float, r1: float,
                     sh: SpatialHash3D, params: 'GrowthParams',
                     samples: int = 8) -> bool:
    """
    Return True if the segment p0->p1 is collision-free.
    """
    if samples < 1:
        samples = 1
    skip = float(params.skip_t_head)
    for i in range(samples + 1):
        t = skip + (1.0 - skip) * (i / float(samples))
        x = float(p0[0] + t * (p1[0] - p0[0]))
        y = float(p0[1] + t * (p1[1] - p0[1]))
        z = float(p0[2] + t * (p1[2] - p0[2]))
        r = float(r0 + t * (r1 - r0))
        search_r = r + params.surface_clearance + 2.0 * sh.cell
        for nx, ny, nz, nr in sh.nearby(x, y, z, search_r):
            dx = x - nx; dy = y - ny; dz = z - nz
            d2 = dx*dx + dy*dy + dz*dz
            thresh = _clearance_threshold(
                r, float(nr),
                mode=params.clearance_mode,
                cf=params.clearance_factor,
                cap=params.clearance_cap,
                surface=params.surface_clearance
            )
            if d2 <= (thresh * thresh):
                return False
    return True

# -------------------------- Tree growth model --------------------------- #

@dataclass
class GrowthParams:
    shape: Tuple[int, int, int]               # (X, Y, Z)
    min_diam: float = 3.0
    max_diam: float = 250.0
    initial_diam: Optional[float] = None      # override initial trunk diameter
    min_seg_len: float = 12.0
    max_seg_len: float = 80.0
    tip_perturb_angle_deg: float = 18.0       # per-segment small bend
    branch_angle_deg: float = 35.0            # angle between daughters
    branch_prob: float = 0.25                 # chance a tip branches
    taper_factor: float = 0.92                # multiplicative taper along extension
    murray_exp: float = 3.0                   # r0^m = r1^m + r2^m at bifurcations
    min_diam_stop: Optional[float] = None     # clamp terminal diameters
    max_segments: int = 800                   # global cap
    curved: bool = True
    curve_mode: str = "catmull"               # 'bezier' or 'catmull'
    samples_per_seg: int = 8                  # centreline samples per logical segment
    seed_points: int = 1                      # number of roots
    boundary_margin: int = 8                  # keep this many voxels inside the volume
    # Self-avoidance
    self_avoid: bool = True
    clearance_factor: float = 0.9
    surface_clearance: float = 2.0
    sh_cell: float = 16.0
    max_dir_tries: int = 12
    skip_t_head: float = 0.15
    # Tip-hole (omit last samples from spatial hash)
    avoidance_tip_hole_frac: float = 0.25
    avoidance_tip_hole_min: int = 4
    # Complexity (0..1)
    complexity: Optional[float] = None
    # Inward warm-up
    root_inward: bool = False
    first_seg_len: Optional[float] = None
    # Forced branching depth
    force_branch_depth: int = 0
    # Early-branch control
    no_branch_before_depth: int = 0
    early_branch_depth: int = 0
    early_branch_prob: Optional[float] = None
    # Clearance shaping
    clearance_mode: str = "sum"               # 'sum'|'min'|'geom'
    clearance_cap: Optional[float] = None
    # Wall avoidance
    wall_clearance: float = 1.0               # min gap between vessel surface and walls (voxels)
    max_wall_avoid_bend_deg: float = 45.0     # max allowed bend when avoiding walls

def build_samples(raw_polyline: np.ndarray, params: GrowthParams, rng: np.random.Generator) -> np.ndarray:
    if not params.curved:
        p0, p1 = raw_polyline[0], raw_polyline[-1]
        t = np.linspace(0.0, 1.0, max(2, params.samples_per_seg), dtype=np.float32)
        return (p0[None, :] * (1 - t[:, None]) + p1[None, :] * t[:, None]).astype(np.float32)

    if params.curve_mode == 'bezier':
        p0, p1 = raw_polyline[0], raw_polyline[-1]
        dirv = unit(p1 - p0)
        u, v, _ = orthonormal_basis(dirv)
        bend_axis = u if np.random.rand() < 0.5 else v
        bend_mag = np.random.uniform(0.2, 0.6) * np.linalg.norm(p1 - p0)
        c = (p0 + p1) * 0.5 + bend_axis * bend_mag
        return quadratic_bezier(p0, p1, c, max(2, params.samples_per_seg)).astype(np.float32)

    # default Catmull-Rom through 3 jittered points
    p0, p1 = raw_polyline[0], raw_polyline[-1]
    mid = (p0 + p1) / 2.0
    jitter = random_unit_vector(np.random.default_rng()) * np.random.uniform(0.2, 0.6) * np.linalg.norm(p1 - p0)
    pts = np.stack([p0, mid + jitter, p1], axis=0)
    return catmull_rom_spline(pts, max(2, params.samples_per_seg)).astype(np.float32)

# ---------------------- NEW: helpers for tip-hole & wall safety --------- #
def _keep_upto_index(n_samples: int, frac: float, min_tail: int) -> int:
    """Return exclusive end index for insertion (keep samples[0:idx]). Leaves a 'tip hole'."""
    hole = max(int(round(n_samples * float(frac))), int(min_tail))
    return max(0, n_samples - hole)

def _segment_wall_safe(src: np.ndarray, end: np.ndarray, r0: float, r1: float, params: GrowthParams) -> bool:
    """
    Approximate test: ensure that along the straight segment, at several points,
    the vessel surface stays at least wall_clearance away from the walls.
    """
    X, Y, Z = params.shape
    m = params.boundary_margin
    for t in (0.25, 0.5, 0.75, 1.0):
        p = src + t * (end - src)
        r = float(r0 + t * (r1 - r0))
        mx = min(p[0] - m, (X - m) - p[0])
        my = min(p[1] - m, (Y - m) - p[1])
        mz = min(p[2] - m, (Z - m) - p[2])
        if min(mx, my, mz) < (r + params.wall_clearance):
            return False
    return True

def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = unit(a); b = unit(b)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(d))

def generate_graph(params: GrowthParams, rng: np.random.Generator) -> VascularGraph:
    """
    Grow a vascular graph inside a box using randomized tip-extension with optional curvature,
    tapering, self-avoidance, and branching subject to Murray's law.
    """
    X, Y, Z = params.shape
    graph = VascularGraph()

    # Clamp stop diameter
    min_stop = params.min_diam if params.min_diam_stop is None else params.min_diam_stop

    # Spatial hash for self-avoidance
    sh = SpatialHash3D(cell_size=params.sh_cell)

    # Place root seeds
    roots = []
    center = np.array([X * 0.5, Y * 0.5, Z * 0.5], dtype=np.float32)
    for _ in range(params.seed_points):
        x = rng.uniform(params.boundary_margin, X - params.boundary_margin)
        y = rng.uniform(params.boundary_margin, Y - params.boundary_margin)
        z = rng.uniform(params.boundary_margin, Z * 0.2)  # near bottom-ish
        init_diam = params.initial_diam if params.initial_diam is not None else rng.uniform(params.min_diam, params.max_diam)
        nid = graph.add_node(np.array([x, y, z], dtype=np.float32), radius=init_diam * 0.5)
        roots.append(nid)
    graph.roots = roots

    # Tips, with depth tracking (depth = number of accepted extension steps from root)
    tips: List[Tuple[int, np.ndarray]] = []
    tip_depth: Dict[int, int] = {}

    for nid in roots:
        src = np.array(graph.nodes[nid].xyz, dtype=np.float32)
        if params.root_inward:
            dir0 = unit(center - src)  # point inward
            warm_len = params.first_seg_len
            if warm_len is None:
                warm_len = max(params.min_seg_len * 2.0, 0.06 * float(min(X, Y, Z)))  # ~6% of volume size
            end = src + dir0 * warm_len
            end[0] = np.clip(end[0], params.boundary_margin, X - params.boundary_margin)
            end[1] = np.clip(end[1], params.boundary_margin, Y - params.boundary_margin)
            end[2] = np.clip(end[2], params.boundary_margin, Z - params.boundary_margin)
            r_start = graph.nodes[nid].radius
            r_end = r_start * params.taper_factor
            raw = np.stack([src, end], axis=0)
            samples = build_samples(raw, params, rng)
            radii = np.linspace(r_start, r_end, len(samples)).astype(np.float32)
            end_id = graph.add_node(end, radius=r_end)
            graph.add_edge(nid, end_id, samples=samples.tolist(), radii=radii.tolist())
            k = _keep_upto_index(len(samples), params.avoidance_tip_hole_frac, params.avoidance_tip_hole_min)
            if k > 0:
                sh.insert_polyline(samples[:k], radii[:k])
            tips.append((end_id, dir0))
            tip_depth[end_id] = 1
        else:
            dir0 = random_unit_vector(rng); dir0[2] = abs(dir0[2]); dir0 = unit(dir0)
            tips.append((nid, dir0))
            tip_depth[nid] = 0

    def accept_and_record_edge(start_id: int, end_xyz: np.ndarray, r_start: float, r_end: float,
                               direction: np.ndarray) -> Tuple[int, np.ndarray]:
        end_id = graph.add_node(end_xyz, radius=r_end)
        raw = np.stack([np.array(graph.nodes[start_id].xyz, dtype=np.float32), end_xyz], axis=0)
        samples = build_samples(raw, params, rng)
        radii = np.linspace(r_start, r_end, len(samples)).astype(np.float32)
        graph.add_edge(start_id, end_id, samples=samples.tolist(), radii=radii.tolist())
        k = _keep_upto_index(len(samples), params.avoidance_tip_hole_frac, params.avoidance_tip_hole_min)
        if k > 0:
            sh.insert_polyline(samples[:k], radii[:k])
        return end_id, direction

    segments_created = 0
    while tips and segments_created < params.max_segments:
        new_tips: List[Tuple[int, np.ndarray]] = []
        new_depths: Dict[int, int] = {}
        rng.shuffle(tips)
        for nid, direction in tips:
            if segments_created >= params.max_segments:
                break
            src = np.array(graph.nodes[nid].xyz, dtype=np.float32)
            radius = graph.nodes[nid].radius
            depth = tip_depth.get(nid, 0)
            diam = radius * 2.0
            if diam < min_stop:
                continue

            # try to find a collision-free + wall-safe segment
            attempt = 0
            accepted = False
            chosen_end = None
            base_dir = np.array(direction, dtype=np.float32)
            chosen_dir = base_dir
            while attempt < params.max_dir_tries and not accepted:
                seg_len = rng.uniform(params.min_seg_len, params.max_seg_len)
                test_dir = rotate_towards(chosen_dir, math.radians(params.tip_perturb_angle_deg), rng)
                end = src + unit(test_dir) * seg_len
                end[0] = np.clip(end[0], params.boundary_margin, X - params.boundary_margin)
                end[1] = np.clip(end[1], params.boundary_margin, Y - params.boundary_margin)
                end[2] = np.clip(end[2], params.boundary_margin, Z - params.boundary_margin)

                wall_ok = _segment_wall_safe(src, end, radius, radius * params.taper_factor, params)
                avoid_ok = (not params.self_avoid) or segment_is_clear(
                    src, end, radius, radius * params.taper_factor, sh,
                    params, samples=params.samples_per_seg)

                if wall_ok and avoid_ok:
                    accepted = True
                    chosen_end = end
                    chosen_dir = unit(test_dir)
                    break

                # if avoiding a wall would require too much bend, stop this tip
                total_bend = _angle_deg(test_dir, base_dir)
                if (not wall_ok) and (total_bend > params.max_wall_avoid_bend_deg):
                    accepted = False
                    break

                attempt += 1
                chosen_dir = test_dir

            if not accepted:
                continue

            # --------- Early-branch gating / schedule ----------
            branching_allowed = (depth >= params.no_branch_before_depth)
            local_branch_prob = params.branch_prob
            if (params.early_branch_prob is not None) and (depth < params.early_branch_depth):
                local_branch_prob = params.early_branch_prob
            force_branch = branching_allowed and (depth < params.force_branch_depth)
            can_size_branch = (diam > params.min_diam * 1.1)
            random_wants_branch = (rng.random() < local_branch_prob)
            will_branch = force_branch or (branching_allowed and random_wants_branch and can_size_branch)
            # ---------------------------------------------------

            if will_branch:
                # Two daughters (Murray's law split)
                f = rng.uniform(0.3, 0.7)
                r0m = radius ** params.murray_exp
                r1 = (f * r0m) ** (1.0 / params.murray_exp)
                r2 = ((1 - f) * r0m) ** (1.0 / params.murray_exp)

                # Parent segment nid -> bifurcation
                bif_id, _ = accept_and_record_edge(nid, chosen_end, r_start=radius, r_end=radius * params.taper_factor,
                                                   direction=chosen_dir)

                # Child directions
                base_dir2 = chosen_dir
                u, v, _ = orthonormal_basis(base_dir2)
                a1 = unit(rotate_vector(base_dir2, u, math.radians(params.branch_angle_deg)))
                a2 = unit(rotate_vector(base_dir2, v, math.radians(params.branch_angle_deg)))

                # Tiny connectors from bifurcation
                link = 2.0
                nid1 = graph.add_node(chosen_end + a1 * 1e-3, radius=r1)
                nid2 = graph.add_node(chosen_end + a2 * 1e-3, radius=r2)
                for child_id, avec, rr in [(nid1, a1, r1), (nid2, a2, r2)]:
                    raw = np.stack([chosen_end, chosen_end + avec * link], axis=0)
                    samples = build_samples(raw, params, rng)
                    radii = np.linspace(radius * params.taper_factor, rr, len(samples)).astype(np.float32)
                    graph.add_edge(bif_id, child_id, samples=samples.tolist(), radii=radii.tolist())
                    k = _keep_upto_index(len(samples), params.avoidance_tip_hole_frac, params.avoidance_tip_hole_min)
                    if k > 0:
                        sh.insert_polyline(samples[:k], radii[:k])

                new_tips.append((nid1, a1))
                new_tips.append((nid2, a2))
                new_depths[nid1] = depth + 1
                new_depths[nid2] = depth + 1
                segments_created += 1
            else:
                # extension with taper (no split)
                new_radius = radius * params.taper_factor
                end_id, chosen_dir = accept_and_record_edge(nid, chosen_end, r_start=radius, r_end=new_radius,
                                                            direction=chosen_dir)
                new_tips.append((end_id, chosen_dir))
                new_depths[end_id] = depth + 1
                segments_created += 1

        tips = new_tips
        tip_depth = new_depths

    return graph

# ---------------------------- I/O & export ------------------------------ #

def ensure_out(path: str):
    os.makedirs(path, exist_ok=True)

def compute_strahler_orders(graph: VascularGraph) -> Tuple[Dict[int,int], Dict[int,int]]:
    """Compute Strahler order for nodes and edges on a forest rooted at graph.roots."""
    from collections import deque
    parents: Dict[int, Optional[int]] = {r: None for r in graph.roots}
    children: Dict[int, List[int]] = {nid: [] for nid in graph.nodes}

    dq = deque(graph.roots)
    visited = set(graph.roots)
    while dq:
        u = dq.popleft()
        for v in graph.adj.get(u, []):
            if v in visited:
                continue
            visited.add(v)
            parents[v] = u
            children[u].append(v)
            dq.append(v)

    order: Dict[int,int] = {}
    pending = set(graph.nodes.keys())
    changed = True
    while changed and pending:
        changed = False
        for nid in list(pending):
            if len(children[nid]) == 0:
                order[nid] = 1
                pending.remove(nid)
                changed = True
            elif all((c in order) for c in children[nid]):
                vals = [order[c] for c in children[nid]]
                m = max(vals)
                kmax = sum(1 for v in vals if v == m)
                order[nid] = m + 1 if kmax >= 2 else m
                pending.remove(nid)
                changed = True

    eorder: Dict[int,int] = {}
    for eid, e in graph.edges.items():
        if parents.get(e.end, None) == e.start:
            eorder[eid] = order.get(e.end, 1)
        elif parents.get(e.start, None) == e.end:
            eorder[eid] = order.get(e.start, 1)
        else:
            eorder[eid] = max(order.get(e.start, 1), order.get(e.end, 1))
    return order, eorder

def save_metadata_csv(sample_dir: str, graph: VascularGraph, shape: Tuple[int, int, int]):
    """Write graph metadata CSV files into sample_dir."""
    ensure_out(sample_dir)
    node_strahler, edge_strahler = compute_strahler_orders(graph)

    with open(os.path.join(sample_dir, "nodes.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["node_id", "x", "y", "z", "radius", "diameter", "strahler"])
        for nid, node in graph.nodes.items():
            x, y, z = node.xyz
            r = float(node.radius)
            w.writerow([nid, x, y, z, r, 2.0 * r, node_strahler.get(nid, 1)])

    with open(os.path.join(sample_dir, "edges.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "start", "end", "strahler"])
        for eid, e in graph.edges.items():
            w.writerow([eid, e.start, e.end, edge_strahler.get(eid, 1)])

    with open(os.path.join(sample_dir, "edge_samples.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "sample_idx", "x", "y", "z", "radius", "diameter"])
        for eid, e in graph.edges.items():
            for si, p in enumerate(e.samples):
                x, y, z = p
                r = float(e.radii[si])
                w.writerow([eid, si, x, y, z, r, 2.0 * r])

    with open(os.path.join(sample_dir, "branches.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["branch_id", "start_node_id", "end_node_id", "path_node_ids"])
        for bid, (s, t, path) in enumerate(graph.branch_endpoints()):
            path_str = ";".join(str(n) for n in path)
            w.writerow([bid, s, t, path_str])

    endpoints = set()
    for s, t, _ in graph.branch_endpoints():
        endpoints.add(s); endpoints.add(t)

    with open(os.path.join(sample_dir, "branch_endpoints.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["node_id", "x", "y", "z", "radius", "diameter", "strahler", "degree"])
        for nid in sorted(endpoints):
            node = graph.nodes[nid]
            x, y, z = node.xyz
            r = float(node.radius)
            deg = len(graph.adj.get(nid, []))
            w.writerow([nid, x, y, z, r, 2.0 * r, node_strahler.get(nid, 1), deg])

    with open(os.path.join(sample_dir, "adjacency.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["node_id", "neighbors"])
        for nid, neigh in graph.adj.items():
            w.writerow([nid, ";".join(str(n) for n in neigh)])

# ---------------------- Output formats (Zarr / NIfTI / TIFF) ------------ #

def create_zarr_arrays(sample_dir: str, chunks: Tuple[int, int, int], shape: Tuple[int, int, int]):
    if zarr is None:
        raise RuntimeError("zarr is not installed. Please `pip install zarr numcodecs`.")
    ensure_out(sample_dir)
    grp = zarr.group(store=zarr.DirectoryStore(os.path.join(sample_dir, "volume.zarr")))
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
    vol = grp.require_dataset('vessels', shape=(shape[2], shape[1], shape[0]),
                              chunks=(chunks[2], chunks[1], chunks[0]),
                              dtype='u1', compressor=compressor, overwrite=True, fill_value=0)
    cl = grp.require_dataset('centreline', shape=(shape[2], shape[1], shape[0]),
                             chunks=(chunks[2], chunks[1], chunks[0]),
                             dtype='u1', compressor=compressor, overwrite=True, fill_value=0)
    return vol, cl

def _memmap_close(mm):
    try:
        if hasattr(mm, 'flush'):
            mm.flush()
    except Exception:
        pass
    try:
        # numpy.memmap exposes underlying mmap at _mmap on most platforms
        m = getattr(mm, '_mmap', None)
        if m is not None:
            m.close()
    except Exception:
        pass

def _finalize_and_cleanup_memmaps(v_mm, c_mm, tmp_dir: str, out_dir: str, pixdim: Tuple[float,float,float], *,
                                  write_nifti: bool = True, write_tiff: bool = False, Z: Optional[int] = None):
    # Save images if requested
    if write_nifti:
        if nib is None:
            raise RuntimeError("nibabel is required for NIfTI output.")
        sx, sy, sz = pixdim
        aff = np.diag([sx, sy, sz, 1.0]).astype(np.float32)
        img_v = nib.Nifti1Image(np.asarray(v_mm), affine=aff)
        img_c = nib.Nifti1Image(np.asarray(c_mm), affine=aff)
        nib.save(img_v, os.path.join(out_dir, "vessels.nii.gz"))
        nib.save(img_c, os.path.join(out_dir, "centreline.nii.gz"))
    if write_tiff:
        if tifffile is None:
            raise RuntimeError("tifffile is required for TIFF output.")
        assert Z is not None
        vess_dir = os.path.join(out_dir, "tiff", "vessels")
        cent_dir = os.path.join(out_dir, "tiff", "centreline")
        ensure_out(vess_dir); ensure_out(cent_dir)
        for z in range(Z):
            tifffile.imwrite(os.path.join(vess_dir, f"z{z:05d}.tif"), np.asarray(v_mm[z]))
            tifffile.imwrite(os.path.join(cent_dir, f"z{z:05d}.tif"), np.asarray(c_mm[z]))

    # Close and drop memmaps before deleting directory
    try:
        _memmap_close(v_mm); _memmap_close(c_mm)
    except Exception:
        pass
    try:
        del v_mm; del c_mm
    except Exception:
        pass
    gc.collect()
    # Give Windows a beat to release file locks
    time.sleep(0.05)
    try:
        shutil.rmtree(tmp_dir, ignore_errors=False)
    except Exception:
        # last resort, try again ignoring errors
        shutil.rmtree(tmp_dir, ignore_errors=True)

def create_output_arrays(sample_dir: str, shape: Tuple[int,int,int], chunks: Tuple[int,int,int], fmt: str,
                         pixdim: Tuple[float,float,float]):
    """Create output storages and return (arr_vessels, arr_centreline, finalize_fn)."""
    fmt = fmt.lower()
    Z, Y, X = shape[2], shape[1], shape[0]
    if fmt == 'zarr':
        vol, cl = create_zarr_arrays(sample_dir, chunks, shape)
        def finalize():
            return None
        return vol, cl, finalize

    # For nifti and tiff, use memmaps (tmp inside the sample dir)
    tmp_dir = os.path.join(sample_dir, "tmp")
    ensure_out(tmp_dir)
    v_path = os.path.join(tmp_dir, 'vessels.dat')
    c_path = os.path.join(tmp_dir, 'centreline.dat')
    vol_mm = np.memmap(v_path, dtype=np.uint8, mode='w+', shape=(Z, Y, X))
    cl_mm = np.memmap(c_path, dtype=np.uint8, mode='w+', shape=(Z, Y, X))
    vol_mm[:] = 0
    cl_mm[:] = 0

    if fmt == 'nifti':
        def finalize():
            _finalize_and_cleanup_memmaps(vol_mm, cl_mm, tmp_dir, sample_dir, pixdim,
                                          write_nifti=True, write_tiff=False)
        return vol_mm, cl_mm, finalize

    if fmt == 'tiff_slices':
        def finalize():
            _finalize_and_cleanup_memmaps(vol_mm, cl_mm, tmp_dir, sample_dir, pixdim,
                                          write_nifti=False, write_tiff=True, Z=Z)
        return vol_mm, cl_mm, finalize

    raise ValueError(f"Unknown output format: {fmt}. Choose from 'zarr', 'nifti', 'tiff_slices'.")

def write_volumes(sample_dir: str, graph: VascularGraph, shape: Tuple[int, int, int], chunks: Tuple[int, int, int],
                  output_format: str = 'zarr', pixdim: Tuple[float,float,float] = (1.0,1.0,1.0)):
    vol, cl, finalize = create_output_arrays(sample_dir, shape, chunks, output_format, pixdim)
    for eid in tqdm(list(graph.edges.keys()), desc=f"painting {os.path.basename(sample_dir)}"):
        e = graph.edges[eid]
        paint_edge_to_zarr(vol, cl, e, chunk=(chunks[2], chunks[1], chunks[0]), shape=(shape[0], shape[1], shape[2]))
    finalize()

# ---------------------- Binned NIfTI exports (re-rasterize) ------------- #

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def _create_binned_nifti_pair(sample_dir: str,
                              shape_xyz: Tuple[int, int, int],
                              pixdim_xyz: Tuple[float, float, float],
                              factor: int):
    """
    Create memmaps for a binned (downsampled) NIfTI pair. Returns (vol_mm, cl_mm, finalize_fn).
    Filenames: vessels_ds{factor}.nii.gz, centreline_ds{factor}.nii.gz
    """
    if nib is None:
        raise RuntimeError("nibabel is required for --binned-export. Please `pip install nibabel`.")

    Xs, Ys, Zs = shape_xyz
    Z, Y, X = Zs, Ys, Xs  # memory order (Z,Y,X)

    tmp_dir = os.path.join(sample_dir, f"tmp_ds{factor}")
    ensure_out(tmp_dir)
    v_path = os.path.join(tmp_dir, 'vessels.dat')
    c_path = os.path.join(tmp_dir, 'centreline.dat')

    vol_mm = np.memmap(v_path, dtype=np.uint8, mode='w+', shape=(Z, Y, X))
    cl_mm  = np.memmap(c_path, dtype=np.uint8, mode='w+', shape=(Z, Y, X))
    vol_mm[:] = 0
    cl_mm[:]  = 0

    out_v = os.path.join(sample_dir, f"vessels_ds{factor}.nii.gz")
    out_c = os.path.join(sample_dir, f"centreline_ds{factor}.nii.gz")

    def finalize():
        nonlocal vol_mm, cl_mm  # <-- important: avoid UnboundLocalError
        sx, sy, sz = pixdim_xyz
        aff = np.diag([sx, sy, sz, 1.0]).astype(np.float32)

        # Save while memmaps are still open
        img_v = nib.Nifti1Image(np.asarray(vol_mm), affine=aff)
        img_c = nib.Nifti1Image(np.asarray(cl_mm),  affine=aff)
        nib.save(img_v, out_v)
        nib.save(img_c, out_c)

        # Close and release file handles
        try:
            _memmap_close(vol_mm); _memmap_close(cl_mm)
        finally:
            vol_mm = None
            cl_mm = None

        gc.collect()
        time.sleep(0.05)  # let Windows release locks
        try:
            shutil.rmtree(tmp_dir, ignore_errors=False)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return vol_mm, cl_mm, finalize


def write_binned_nifti_exports(sample_dir: str,
                               graph: VascularGraph,
                               base_shape: Tuple[int, int, int],    # (X,Y,Z)
                               base_pixdim: Tuple[float, float, float],
                               bin_exponents: Optional[List[int]]):
    """
    Re-rasterize the graph at powers-of-two downsampled grids and emit NIfTI pairs
    (vessels + centreline) with topology preserved.
    Example: bin_exponents=[1,2] -> ds2 and ds4.
    """
    if not bin_exponents:
        return

    X, Y, Z = base_shape
    sx, sy, sz = base_pixdim
    exps = sorted(set(int(e) for e in bin_exponents if int(e) > 0))
    if not exps:
        return

    for e in exps:
        s = 1 << e  # factor = 2**e
        Xs = _ceil_div(X, s)
        Ys = _ceil_div(Y, s)
        Zs = _ceil_div(Z, s)
        pixdim_s = (sx * s, sy * s, sz * s)

        vol, cl, finalize = _create_binned_nifti_pair(sample_dir,
                                                      (Xs, Ys, Zs),
                                                      pixdim_s,
                                                      factor=s)

        # choose a conservative chunk for painting
        cz = min(128, Zs); cy = min(128, Ys); cx = min(128, Xs)

        # paint each edge at scaled coordinates/radii
        for eid, eedge in tqdm(list(graph.edges.items()), desc=f"painting ds{s}"):
            pts = np.asarray(eedge.samples, dtype=np.float32) / float(s)
            r   = np.asarray(eedge.radii,  dtype=np.float32) / float(s)
            e_scaled = Edge(id=eid, start=-1, end=-1,
                            samples=[(float(px), float(py), float(pz)) for (px,py,pz) in pts],
                            radii=[float(rv) for rv in r])
            paint_edge_to_zarr(vol, cl, e_scaled,
                               chunk=(cz, cy, cx),
                               shape=(Xs, Ys, Zs))

        finalize()

# ------------------------------ CLI / Main ------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic vascular volumes for skeletonization training.")
    p.add_argument('--out', type=str, required=True, help='Output directory for dataset')
    p.add_argument('--num', type=int, default=1, help='Number of samples to generate')
    p.add_argument('--shape', type=int, nargs=3, metavar=('X','Y','Z'), default=[512, 512, 512],
                   help='Volume shape in voxels (X Y Z). Up to 3000 3000 3000 supported with Zarr.')
    p.add_argument('--chunks', type=int, nargs=3, metavar=('CX','CY','CZ'), default=[160, 160, 160],
                   help='Chunk size in voxels for Zarr writing (X Y Z).')
    p.add_argument('--min-diam', type=float, default=3.0, help='Minimum vessel diameter (voxels)')
    p.add_argument('--max-diam', type=float, default=250.0, help='Maximum vessel diameter (voxels)')
    p.add_argument('--initial-diam', type=float, default=None,
                   help='Fixed initial root diameter (voxels); overrides random sampling')
    p.add_argument('--min-seg', type=float, default=12.0, help='Minimum logical segment length (voxels)')
    p.add_argument('--max-seg', type=float, default=80.0, help='Maximum logical segment length (voxels)')
    p.add_argument('--branch-prob', type=float, default=0.25, help='Branch probability per tip')
    p.add_argument('--taper', type=float, default=0.92, help='Per-segment taper factor (<1)')
    p.add_argument('--murray-exp', type=float, default=3.0, help="Murray's law exponent (r0^m = r1^m + r2^m)")
    p.add_argument('--max-segments', type=int, default=800, help='Global cap on number of segments')
    p.add_argument('--curved', action='store_true', help='Enable curved branches')
    p.add_argument('--curve-mode', type=str, default='catmull', choices=['catmull', 'bezier'], help='Curve type')
    p.add_argument('--samples-per-seg', type=int, default=8, help='Centreline samples per segment (>=2)')
    p.add_argument('--seeds', type=int, default=1, help='Number of root seeds')
    p.add_argument('--seed', type=int, default=0, help='Random seed (0 means random)')
    p.add_argument('--boundary', type=int, default=8, help='Boundary margin (voxels)')
    # Self-avoidance
    p.add_argument('--self-avoid', action='store_true', help='Enable spatial self-avoidance checks')
    p.add_argument('--clearance-factor', type=float, default=0.9, help='Multiplier on (r1+r2) in collision threshold')
    p.add_argument('--surface-clearance', type=float, default=2.0,
                   help='Extra surface-to-surface clearance (voxels) added to collision threshold')
    p.add_argument('--sh-cell', type=float, default=16.0, help='Cell size for spatial hash (voxels)')
    p.add_argument('--max-dir-tries', type=int, default=12, help='Attempts to rotate a tip to avoid collisions')
    p.add_argument('--skip-t-head', type=float, default=0.15,
                   help='Fraction of each new segment to skip at the tip during clearance checks')
    # Tip-hole tuning
    p.add_argument('--avoidance-tip-hole-frac', type=float, default=0.25,
                   help='Fraction of samples at the end of each new edge NOT inserted in the spatial hash (tip-hole).')
    p.add_argument('--avoidance-tip-hole-min', type=int, default=4,
                   help='Minimum number of samples to omit at the tip for the tip-hole.')
    p.add_argument('--clearance-mode', type=str, default='sum',
                   choices=['sum','min','geom'],
                   help="Distance basis for avoidance: 'sum'=(r+nr)*cf, 'min'=min(r,nr)*cf, 'geom'=sqrt(r*nr)*cf.")
    p.add_argument('--clearance-cap', type=float, default=None,
                   help='Optional cap (voxels) on the clearance term (before adding surface_clearance).')
    # Wall avoidance
    p.add_argument('--wall-clearance', type=float, default=1.0,
                   help='Min gap (voxels) required between vessel surface and box walls along a new segment.')
    p.add_argument('--max-wall-avoid-bend', type=float, default=45.0,
                   help='If avoiding a wall would require turning more than this angle (deg), stop the tip.')
    # Complexity control
    p.add_argument('--complexity', type=float, default=None, help='0..1 factor scaling density/branching')
    # Inward warm-up
    p.add_argument('--root-inward', action='store_true',
                   help='Aim the first segment(s) from each root toward the volume centre')
    p.add_argument('--first-seg-len', type=float, default=None,
                   help='Length of the initial inward segment (voxels). Default ~6% of min(X,Y,Z)')
    # Terminal stop
    p.add_argument('--min-diam-stop', type=float, default=None,
                   help='Terminal diameter threshold to stop growth (voxels). Default = min-diam')
    # Forced branching
    p.add_argument('--force-branch-depth', type=int, default=0,
                   help='Force branching for tips whose depth from a root is < this value')
    # Early-branch control
    p.add_argument('--no-branch-before-depth', type=int, default=0, metavar='K',
                   help='Forbid any branching while depth < K (counted in accepted segments from the root).')
    p.add_argument('--early-branch-depth', type=int, default=0, metavar='D',
                   help='While depth < D, override branch probability with --early-branch-prob (if set). 0 disables.')
    p.add_argument('--early-branch-prob', type=float, default=None, metavar='P',
                   help='Branch probability P used while depth < --early-branch-depth. Omit to use normal --branch-prob.')
    # Output formats
    p.add_argument('--format', type=str, default='zarr',
                choices=['zarr','nifti','tiff_slices'],
                help='Primary output format for volumes')
    p.add_argument('--also-nifti', action='store_true',
                help='In addition to --format, also write NIfTI volumes (vessels.nii.gz, centreline.nii.gz).')
    p.add_argument('--also-tiff-slices', action='store_true',
                help='In addition to --format, also write TIFF slice stacks (tiff/vessels/, tiff/centreline/).')
    p.add_argument('--pixdim', type=float, nargs=3, metavar=('SX','SY','SZ'), default=[1.0,1.0,1.0],
                help='Voxel spacing for NIfTI (mm)')
    # Binned NIfTI export (powers of 2)
    p.add_argument('--binned-export', type=int, nargs='*', default=None, metavar='K',
                help='Export extra NIfTIs downsampled by 2^K (via re-rasterization), e.g. "--binned-export 1 2" for /2 and /4.')
    # Workers
    p.add_argument('--workers', type=int, default=0, help='Worker processes for parallel sample generation (0=auto)')
    return p.parse_args()

def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return 'na'
    s = f"{x}"
    return s.replace('.', 'p')

def build_sample_dir_name(idx: int, params: GrowthParams, fmt: str, pixdim: Tuple[float,float,float], seed: Optional[int]) -> str:
    X, Y, Z = params.shape
    sx, sy, sz = pixdim
    name = (
        f"sample_{idx:04d}"
        f"__shape-{X}x{Y}x{Z}"
        f"__fmt-{fmt}"
        f"__pix-{_fmt_num(sx)}x{_fmt_num(sy)}x{_fmt_num(sz)}"
        f"__curved-{1 if params.curved else 0}"
        f"__comp-{_fmt_num(params.complexity)}"
        f"__seed-{seed if seed is not None else 'rand'}"
    )
    return name

def _generate_and_write(idx: int, out: str, shape: Tuple[int,int,int], chunks: Tuple[int,int,int],
                        gp_kwargs: dict, seed: Optional[int], out_format: str,
                        pixdim: Tuple[float,float,float],
                        also_nifti: bool, also_tiff_slices: bool,
                        binned_pows: Optional[List[int]]):
    rng = np.random.default_rng(seed)
    gp = GrowthParams(**gp_kwargs)
    gp.min_diam_stop = gp.min_diam if gp.min_diam_stop is None else gp.min_diam_stop

    # If extras are requested, tag sample dir as "multi" to avoid misleading __fmt- in the name
    name_fmt = 'multi' if (also_nifti or also_tiff_slices or (binned_pows and len(binned_pows) > 0)) else out_format
    sample_dirname = build_sample_dir_name(idx, gp, name_fmt, pixdim, seed)
    sample_dir = os.path.join(out, sample_dirname)
    ensure_out(sample_dir)

    graph = generate_graph(gp, rng)
    save_metadata_csv(sample_dir, graph, shape)

    # Primary format
    write_volumes(sample_dir, graph, shape, chunks, output_format=out_format, pixdim=pixdim)

    # Extra formats (skip if same as primary)
    if also_nifti and out_format != 'nifti':
        write_volumes(sample_dir, graph, shape, chunks, output_format='nifti', pixdim=pixdim)
    if also_tiff_slices and out_format != 'tiff_slices':
        write_volumes(sample_dir, graph, shape, chunks, output_format='tiff_slices', pixdim=pixdim)

    # Binned NIfTI exports (topology-preserving re-rasterization)
    write_binned_nifti_exports(sample_dir,
                               graph,
                               base_shape=shape,
                               base_pixdim=pixdim,
                               bin_exponents=binned_pows)

    return idx

def main():
    args = parse_args()
    ensure_out(args.out)

    shape = (int(args.shape[0]), int(args.shape[1]), int(args.shape[2]))
    chunks = (int(args.chunks[0]), int(args.chunks[1]), int(args.chunks[2]))

    # Base params from CLI
    gp = GrowthParams(
        shape=shape,
        min_diam=float(args.min_diam),
        max_diam=float(args.max_diam),
        initial_diam=args.initial_diam,
        min_seg_len=float(args.min_seg),
        max_seg_len=float(args.max_seg),
        branch_prob=float(args.branch_prob),
        taper_factor=float(args.taper),
        murray_exp=float(args.murray_exp),
        max_segments=int(args.max_segments),
        curved=bool(args.curved),
        curve_mode=str(args.curve_mode),
        samples_per_seg=int(args.samples_per_seg),
        seed_points=int(args.seeds),
        boundary_margin=int(args.boundary),
        self_avoid=bool(args.self_avoid),
        clearance_factor=float(args.clearance_factor),
        surface_clearance=float(args.surface_clearance),
        sh_cell=float(args.sh_cell),
        max_dir_tries=int(args.max_dir_tries),
        skip_t_head=float(args.skip_t_head),
        avoidance_tip_hole_frac=float(args.avoidance_tip_hole_frac),
        avoidance_tip_hole_min=int(args.avoidance_tip_hole_min),
        complexity=args.complexity,
        root_inward=bool(args.root_inward),
        first_seg_len=args.first_seg_len,
        min_diam_stop=args.min_diam_stop,
        force_branch_depth=int(args.force_branch_depth),
        no_branch_before_depth = int(args.no_branch_before_depth),
        early_branch_depth=int(args.early_branch_depth),
        early_branch_prob=(None if args.early_branch_prob is None else float(args.early_branch_prob)),
        clearance_mode=str(args.clearance_mode),
        clearance_cap=(None if args.clearance_cap is None else float(args.clearance_cap)),
        wall_clearance=float(args.wall_clearance),
        max_wall_avoid_bend_deg=float(args.max_wall_avoid_bend),
    )

    # Optional density/complexity scaling
    if gp.complexity is not None:
        c = float(np.clip(gp.complexity, 0.0, 1.0))
        gp.branch_prob = 0.10 + 0.40 * c
        gp.max_segments = int(gp.max_segments * (0.8 + 1.4 * c))
        gp.min_seg_len = max(4.0, gp.min_seg_len * (1.0 - 0.4 * c))
        gp.max_seg_len = max(gp.min_seg_len+2.0, gp.max_seg_len * (1.0 - 0.3 * c))
        base_ang = 14.0
        gp.tip_perturb_angle_deg = base_ang + 10.0 * c

    # Worker count
    auto_workers = max(1, (os.cpu_count() or 2) - 2)
    workers = auto_workers if int(args.workers) == 0 else int(args.workers)

    # Per-sample seeds
    base_seed = None if int(args.seed) == 0 else int(args.seed)
    rng = np.random.default_rng(base_seed)
    seeds = [None if base_seed is None else int(rng.integers(0, 2**32 - 1)) for _ in range(args.num)]

    gp_kwargs = asdict(gp)
    gp_kwargs['shape'] = tuple(gp_kwargs['shape'])

    binned_pows = None if args.binned_export is None else [int(k) for k in args.binned_export]

    if workers == 1 or args.num == 1:
        for i in range(args.num):
            _generate_and_write(i, args.out, shape, chunks, gp_kwargs, seeds[i],
                                args.format, tuple(args.pixdim),
                                also_nifti=bool(args.also_nifti),
                                also_tiff_slices=bool(args.also_tiff_slices),
                                binned_pows=binned_pows)
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as exe:
            futs = [
                exe.submit(_generate_and_write, i, args.out, shape, chunks, gp_kwargs, seeds[i],
                           args.format, tuple(args.pixdim),
                           bool(args.also_nifti), bool(args.also_tiff_slices),
                           binned_pows)
                for i in range(args.num)
            ]
            for _ in tqdm(cf.as_completed(futs), total=len(futs), desc='samples'):
                _.result()

    print(f"\nDone. Wrote {args.num} sample(s) to: {args.out}")

if __name__ == '__main__':
    main()
