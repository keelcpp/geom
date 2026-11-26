# keel::geom

2D/ND computational geometry, scattered-data interpolation, and linear assignment primitives built on [`keel::keel`](https://github.com/keelcpp/keel).

## Features

- **Polygons**: `polyarea` (shoelace area), `inpolygon_point`, `inpolygon` (ray-casting with exact edge/vertex boundary inclusion).
- **Convex Hull & Alpha-Shape Boundary**: `convhull` (Andrew's monotone chain), `boundary` (shrink-factor Delaunay boundary extraction).
- **2D Delaunay Triangulation**: `delaunay` (scale-invariant Bowyer–Watson incremental triangulation with co-circular and duplicate-point handling).
- **Scattered Interpolation**: `griddata` (Delaunay barycentric `Linear`, `Nearest`, Sibson `Natural` neighbor via Watson circumcenter areas, Clough–Tocher C1 `Cubic`, and Sandwell `BiharmonicV4` spline) and `griddatan_nearest` (ND nearest-neighbor).
- **2D Histogram Binning**: `histcounts2`.
- **Linear Assignment / Bipartite Matching**: `matchpairs` (Jonker–Volgenant Hungarian algorithm with dynamic cost scaling, `Inf`/`NaN` guards, and min/max modes).
- Zero STL in public headers, compiles cleanly under `-fno-exceptions -fno-rtti`.
- Differential reference test suite (`tests/verify_reference.py`) validated against NumPy, SciPy, and MATLAB.
