# keel::geom

2D/ND computational geometry, scattered-data interpolation, and linear assignment primitives built on [`keel::keel`](https://github.com/keelcpp/keel).

## Features

- **Polygons**: `polyarea` (shoelace area), `inpolygon_point`, `inpolygon` (ray-casting).
- **Convex Hull & Alpha-Shape Boundary**: `convhull` (Andrew's monotone chain), `boundary` (shrink-factor Delaunay boundary extraction).
- **2D Delaunay Triangulation**: `delaunay` (exact circumcircle for small clouds + Bowyer-Watson incremental triangulation for larger point sets).
- **Scattered Interpolation**: `griddata` (Delaunay barycentric linear, nearest-neighbor, Sandwell biharmonic spline v4 with coincident-point deduplication) and `griddatan_nearest` (ND nearest-neighbor).
- **2D Histogram Binning**: `histcounts2`.
- **Linear Assignment / Bipartite Matching**: `matchpairs` (Jonker-Volgenant Hungarian algorithm with optional unmatched cost and min/max modes).
- Zero STL in public headers, compiles cleanly under `-fno-exceptions -fno-rtti`.
