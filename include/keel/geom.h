#pragma once

#include "keel/array.h"
#include "keel/core.h"
#include "keel/memory.h"
#include "keel/ref.h"

namespace keel {

// ============================================================================
// 1. 2D Polygons
// ============================================================================

/// Unsigned area of a 2D polygon via the shoelace formula (returns 0 if n < 3).
[[nodiscard]] f64 polyarea(ArrayRef<f64> x, ArrayRef<f64> y) noexcept;

/// Point-in-polygon ray-casting test for a single query point (xq, yq).
[[nodiscard]] bool inpolygon_point(f64 xq, f64 yq, ArrayRef<f64> xv, ArrayRef<f64> yv) noexcept;

/// Batch point-in-polygon test: writes 1 (inside) or 0 (outside) to `out`.
void inpolygon(ArrayRef<f64> xq, ArrayRef<f64> yq, ArrayRef<f64> xv, ArrayRef<f64> yv,
               MutArrayRef<u8> out) noexcept;

// ============================================================================
// 2. Convex Hull, Shrink Boundary & 2D Delaunay Triangulation
// ============================================================================

/// 2D convex hull via Andrew's monotone chain.
/// Returns 0-based vertex indices in CCW order with the start vertex repeated at the end.
[[nodiscard]] Array<usize> convhull(ArrayRef<f64> x, ArrayRef<f64> y, Allocator* alloc = nullptr);

/// 2D shrink-factor boundary (shrink in [0, 1]; 0 => convex hull, 1 => tightest boundary).
/// Returns 0-based vertex indices forming a closed loop.
[[nodiscard]] Array<usize> boundary(ArrayRef<f64> x, ArrayRef<f64> y, f64 shrink = 0.5,
                                    Allocator* alloc = nullptr);

struct Triangle {
    usize v0 = 0;
    usize v1 = 0;
    usize v2 = 0;
};

/// 2D Delaunay triangulation returning 0-based CCW vertex index triples.
[[nodiscard]] Array<Triangle> delaunay(ArrayRef<f64> x, ArrayRef<f64> y, Allocator* alloc = nullptr);

// ============================================================================
// 3. Scattered Data Interpolation (2D griddata & ND nearest)
// ============================================================================

enum class ScatteredMethod : u8 {
    Linear = 0,
    Nearest = 1,
    Natural = 2,
    Cubic = 3,
    BiharmonicV4 = 4,
};

/// 2D scattered data interpolation at query points (xq, yq).
/// Coincident sample points are averaged automatically. Out-of-hull linear queries yield NaN.
/// Returns false on mismatched input sizes.
[[nodiscard]] bool griddata(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> v, ArrayRef<f64> xq,
                            ArrayRef<f64> yq, MutArrayRef<f64> out,
                            ScatteredMethod method = ScatteredMethod::Linear,
                            Allocator* scratch = nullptr);

/// ND nearest-neighbor scattered interpolation.
/// `x_colmajor` is m x dims (column-major), `v` has length m, `xi_colmajor` is k x dims, `out` has length k.
[[nodiscard]] bool griddatan_nearest(ArrayRef<f64> x_colmajor, usize m, usize dims, ArrayRef<f64> v,
                                     ArrayRef<f64> xi_colmajor, usize k, MutArrayRef<f64> out) noexcept;

// ============================================================================
// 4. 2D Histogram Binning
// ============================================================================

/// Bins 2D samples (x, y) into (xedges.size()-1) x (yedges.size()-1) column-major counts.
/// Rightmost bin edges are inclusive; non-finite or out-of-range samples are ignored.
void histcounts2(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> xedges, ArrayRef<f64> yedges,
                 MutArrayRef<f64> out_colmajor) noexcept;

// ============================================================================
// 5. Linear Assignment / Bipartite Matching (Jonker-Volgenant Hungarian)
// ============================================================================

struct PairIndex {
    usize row = 0;
    usize col = 0;
};

struct MatchpairsOutput {
    Array<PairIndex> matches;
    Array<usize> unmatched_rows;
    Array<usize> unmatched_cols;

    explicit MatchpairsOutput(Allocator* alloc = nullptr) noexcept
        : matches(alloc), unmatched_rows(alloc), unmatched_cols(alloc) {}
};

/// Solves rectangular linear assignment on a column-major (rows x cols) cost matrix
/// with per-row/per-col unmatched penalty `cost_unmatched` (0-based indices in output).
[[nodiscard]] MatchpairsOutput matchpairs(ArrayRef<f64> cost_colmajor, usize rows, usize cols,
                                          f64 cost_unmatched, bool maximize = false,
                                          Allocator* alloc = nullptr);

} // namespace keel
