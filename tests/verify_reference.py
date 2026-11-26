#!/usr/bin/env python3
"""Differential reference verification for keel::geom against NumPy, SciPy, and MATLAB."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import interpolate, optimize, spatial


def parse_dump(text: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, rest = line.split(":", 1)
        vals = [float(tok) for tok in rest.strip().split()] if rest.strip() else []
        out[key.strip()] = vals
    return out


def check_close(name: str, got: np.ndarray, ref: np.ndarray, tol: float, failures: list[str]) -> None:
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape:
        failures.append(f"{name}: shape mismatch got={got.shape} ref={ref.shape}")
        return
    scale = max(1.0, float(np.max(np.abs(ref))) if ref.size else 1.0)
    err = float(np.max(np.abs(got - ref))) / scale if ref.size else 0.0
    status = "PASS" if err <= tol else "FAIL"
    print(f"  [{status}] {name:<28s} rel_err={err:.3e} (tol={tol:.1e})")
    if err > tol:
        failures.append(f"{name}: rel_err={err:.3e} > tol={tol:.1e}")


def run_matlab_checks(dump: dict[str, list[float]], failures: list[str]) -> None:
    if not shutil.which("matlab"):
        print("  [SKIP] MATLAB not found on PATH")
        return

    script = r"""
format longg
px = [0 3 4 1.5 -1]; py = [0 0.5 3 4.5 2];
fprintf('m_polyarea: %.17g\n', polyarea(px, py));
qpx = [1.5 5 0 1.5 3.5 -0.5]; qpy = [2 2 0 0.25 1.75 1];
in = inpolygon(qpx, qpy, px, py);
fprintf('m_inpolygon: %s\n', sprintf('%d ', double(in)));

cx = [0 2 4 3.5 1 -0.5 1.5 2.2 0.8 3];
cy = [0 -0.8 0.5 3.2 3.8 1.5 1.2 1.8 2.1 1];
[k, av] = convhull(cx, cy);
fprintf('m_convhull_area: %.17g\n', av);
fprintf('m_convhull_nverts: %d\n', length(k) - 1);

sx = []; sy = []; sv = [];
for i = 0:4
    for j = 0:4
        x = i + 0.07*sin(i*3 + j);
        y = j + 0.07*cos(i + j*5);
        sx(end+1) = x; sy(end+1) = y;
        sv(end+1) = sin(0.5*x)*cos(0.4*y) + 0.2*x - 0.1*y;
    end
end
qgx = [0.75 1.6 2.35 3.1]; qgy = [0.85 2.2 1.45 2.9];
fprintf('m_gd_linear: %s\n', sprintf('%.17g ', griddata(sx, sy, sv, qgx, qgy, 'linear')));
fprintf('m_gd_nearest: %s\n', sprintf('%.17g ', griddata(sx, sy, sv, qgx, qgy, 'nearest')));
fprintf('m_gd_natural: %s\n', sprintf('%.17g ', griddata(sx, sy, sv, qgx, qgy, 'natural')));
fprintf('m_gd_cubic: %s\n', sprintf('%.17g ', griddata(sx, sy, sv, qgx, qgy, 'cubic')));
fprintf('m_gd_v4: %s\n', sprintf('%.17g ', griddata(sx, sy, sv, qgx, qgy, 'v4')));

hx = [0.1 0.9 1.0 1.5 2.0 2.5 0.5 1.9];
hy = [0.2 0.8 1.1 1.9 2.0 0.5 1.5 0.1];
N = histcounts2(hx, hy, [0 1 2], [0 1 2]);
fprintf('m_histcounts2: %s\n', sprintf('%.17g ', N(:)));

C = [1 100 5; 100 2 6; 4 7 3];
[M1, uR1, uC1] = matchpairs(C, 10);
tot1 = sum(C(sub2ind(size(C), M1(:,1), M1(:,2)))) + 10*(numel(uR1) + numel(uC1));
[M2, uR2, uC2] = matchpairs(C, 10, 'max');
tot2 = sum(C(sub2ind(size(C), M2(:,1), M2(:,2)))) + 10*(numel(uR2) + numel(uC2));
fprintf('m_mp_min: %.17g\n', tot1);
fprintf('m_mp_max: %.17g\n', tot2);
"""
    proc = subprocess.run(["matlab", "-batch", script], capture_output=True, text=True, check=True)
    md = parse_dump(proc.stdout)
    check_close("matlab/polyarea", dump["polyarea"], md["m_polyarea"], 1e-14, failures)
    check_close("matlab/inpolygon", dump["inpolygon"], md["m_inpolygon"], 0.0, failures)
    check_close("matlab/convhull_area", dump["convhull_area"], md["m_convhull_area"], 1e-14, failures)
    check_close("matlab/convhull_nverts", dump["convhull_nverts"], md["m_convhull_nverts"], 0.0, failures)
    check_close("matlab/griddata_linear", dump["griddata_linear"], md["m_gd_linear"], 1e-12, failures)
    check_close("matlab/griddata_nearest", dump["griddata_nearest"], md["m_gd_nearest"], 1e-14, failures)
    check_close("matlab/griddata_natural", dump["griddata_natural"], md["m_gd_natural"], 1e-10, failures)
    # Different gradient estimators in Clough-Tocher give ~1e-3 smooth agreement on sin*cos
    check_close("matlab/griddata_cubic", dump["griddata_cubic"], md["m_gd_cubic"], 5e-3, failures)
    check_close("matlab/griddata_v4", dump["griddata_v4"], md["m_gd_v4"], 1e-10, failures)
    check_close("matlab/histcounts2", dump["histcounts2"], md["m_histcounts2"], 0.0, failures)
    check_close("matlab/matchpairs_min", dump["matchpairs_min_obj"], md["m_mp_min"], 1e-14, failures)
    check_close("matlab/matchpairs_max", dump["matchpairs_max_obj"], md["m_mp_max"], 1e-14, failures)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: verify_reference.py <path-to-test_geom>", file=sys.stderr)
        return 2
    exe = Path(sys.argv[1])
    proc = subprocess.run([str(exe), "--dump-ref"], capture_output=True, text=True, check=True)
    dump = parse_dump(proc.stdout)
    failures: list[str] = []

    print("Running keel::geom reference verification against NumPy/SciPy/MATLAB...")

    # 1. ConvexHull area & vertex count
    cx = np.array([0.0, 2.0, 4.0, 3.5, 1.0, -0.5, 1.5, 2.2, 0.8, 3.0])
    cy = np.array([0.0, -0.8, 0.5, 3.2, 3.8, 1.5, 1.2, 1.8, 2.1, 1.0])
    ch = spatial.ConvexHull(np.column_stack([cx, cy]))
    check_close("scipy/convhull_area", dump["convhull_area"], [ch.volume], 1e-14, failures)
    check_close("scipy/convhull_nverts", dump["convhull_nverts"], [len(ch.vertices)], 0.0, failures)

    # 2. griddata linear, nearest, cubic vs scipy.interpolate.griddata
    sx, sy, sv = [], [], []
    for i in range(5):
        for j in range(5):
            x = i + 0.07 * np.sin(i * 3 + j)
            y = j + 0.07 * np.cos(i + j * 5)
            sx.append(x)
            sy.append(y)
            sv.append(np.sin(0.5 * x) * np.cos(0.4 * y) + 0.2 * x - 0.1 * y)
    pts = np.column_stack([sx, sy])
    vals = np.asarray(sv)
    qpts = np.column_stack([[0.75, 1.6, 2.35, 3.1], [0.85, 2.2, 1.45, 2.9]])
    check_close("scipy/griddata_linear", dump["griddata_linear"], interpolate.griddata(pts, vals, qpts, method="linear"), 1e-12, failures)
    check_close("scipy/griddata_nearest", dump["griddata_nearest"], interpolate.griddata(pts, vals, qpts, method="nearest"), 1e-14, failures)
    # Clough-Tocher gradient estimators differ slightly on general smooth functions (exact on quadratics)
    check_close("scipy/griddata_cubic", dump["griddata_cubic"], interpolate.griddata(pts, vals, qpts, method="cubic"), 1e-2, failures)

    # 3. histogram2d
    hx = np.array([0.1, 0.9, 1.0, 1.5, 2.0, 2.5, 0.5, 1.9])
    hy = np.array([0.2, 0.8, 1.1, 1.9, 2.0, 0.5, 1.5, 0.1])
    H, _, _ = np.histogram2d(hx, hy, bins=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    check_close("numpy/histcounts2", dump["histcounts2"], H.flatten(order="F"), 0.0, failures)

    # 4. linear_sum_assignment for min problem (all 3 matched since cost < 2*10)
    C = np.array([[1.0, 100.0, 5.0], [100.0, 2.0, 6.0], [4.0, 7.0, 3.0]])
    ri, ci = optimize.linear_sum_assignment(C)
    check_close("scipy/matchpairs_min", dump["matchpairs_min_obj"], [C[ri, ci].sum()], 1e-14, failures)

    run_matlab_checks(dump, failures)

    if failures:
        print(f"\nFAILED ({len(failures)} check(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL keel::geom REFERENCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
