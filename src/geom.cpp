#include "keel/geom.h"

#include "keel/algo.h"
#include "keel/arena.h"
#include "keel/math.h"
#include "keel/small_array.h"

namespace keel::geom {

namespace {

struct Edge {
    usize a = 0;
    usize b = 0;

    [[nodiscard]] bool operator==(const Edge& o) const noexcept { return a == o.a && b == o.b; }
    [[nodiscard]] bool operator<(const Edge& o) const noexcept { return a != o.a ? a < o.a : b < o.b; }
};

struct AdjEdge {
    usize to = 0;
    usize id = 0;
};

f64 signed_area2(const f64* x, const f64* y, usize a, usize b, usize c) noexcept {
    return (x[b] - x[a]) * (y[c] - y[a]) - (y[b] - y[a]) * (x[c] - x[a]);
}

f64 in_circle_det(const f64* x, const f64* y, usize a, usize b, usize c, usize p) noexcept {
    const f64 ax = x[a] - x[p];
    const f64 ay = y[a] - y[p];
    const f64 bx = x[b] - x[p];
    const f64 by = y[b] - y[p];
    const f64 cx = x[c] - x[p];
    const f64 cy = y[c] - y[p];
    const f64 a2 = ax * ax + ay * ay;
    const f64 b2 = bx * bx + by * by;
    const f64 c2 = cx * cx + cy * cy;
    return ax * (by * c2 - cy * b2) - ay * (bx * c2 - cx * b2) + a2 * (bx * cy - cx * by);
}

Array<Triangle> delaunay_impl(const f64* x, const f64* y, usize n, Allocator* alloc) {
    Array<Triangle> tris(alloc);
    if (n < 3) return tris;
    tris.reserve(2 * n);

    if (n <= 30) {
        for (usize a = 0; a < n; ++a) {
            for (usize b = a + 1; b < n; ++b) {
                for (usize c = b + 1; c < n; ++c) {
                    const f64 sa2 = signed_area2(x, y, a, b, c);
                    if (keel::abs(sa2) < 1e-15) continue;
                    usize va = a;
                    usize vb = b;
                    usize vc = c;
                    if (sa2 < 0.0) keel::swap(vb, vc);
                    bool ok = true;
                    for (usize p = 0; p < n; ++p) {
                        if (p == va || p == vb || p == vc) continue;
                        if (in_circle_det(x, y, va, vb, vc, p) > 1e-12) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) tris.push_back(Triangle{va, vb, vc});
                }
            }
        }
        return tris;
    }

    // Bowyer-Watson incremental Delaunay triangulation for n > 30.
    ArenaTemp scratch = get_scratch(alloc);
    Allocator* salloc = scratch.allocator();
    Array<f64> bwx(n + 3, salloc);
    Array<f64> bwy(n + 3, salloc);
    f64 xmin = x[0];
    f64 xmax = x[0];
    f64 ymin = y[0];
    f64 ymax = y[0];
    for (usize i = 0; i < n; ++i) {
        bwx[i] = x[i];
        bwy[i] = y[i];
        if (x[i] < xmin) xmin = x[i];
        if (x[i] > xmax) xmax = x[i];
        if (y[i] < ymin) ymin = y[i];
        if (y[i] > ymax) ymax = y[i];
    }
    const f64 dmax = keel::max(keel::max(xmax - xmin, ymax - ymin), 1.0);
    const f64 xmid = 0.5 * (xmin + xmax);
    const f64 ymid = 0.5 * (ymin + ymax);
    bwx[n] = xmid - 20.0 * dmax;
    bwy[n] = ymid - dmax;
    bwx[n + 1] = xmid + 20.0 * dmax;
    bwy[n + 1] = ymid - dmax;
    bwx[n + 2] = xmid;
    bwy[n + 2] = ymid + 20.0 * dmax;

    Array<Triangle> cur(salloc);
    Array<Triangle> next_tris(salloc);
    Array<Edge> edges(salloc);
    cur.reserve(2 * n + 4);
    next_tris.reserve(2 * n + 4);
    cur.push_back(Triangle{n, n + 1, n + 2});

    for (usize p = 0; p < n; ++p) {
        edges.clear();
        next_tris.clear();
        for (const Triangle& t : cur) {
            if (in_circle_det(bwx.data(), bwy.data(), t.v0, t.v1, t.v2, p) > 1e-12) {
                edges.push_back(Edge{t.v0, t.v1});
                edges.push_back(Edge{t.v1, t.v2});
                edges.push_back(Edge{t.v2, t.v0});
            } else {
                next_tris.push_back(t);
            }
        }
        for (usize i = 0; i < edges.size(); ++i) {
            bool shared = false;
            for (usize j = 0; j < edges.size(); ++j) {
                if (i != j && edges[i].a == edges[j].b && edges[i].b == edges[j].a) {
                    shared = true;
                    break;
                }
            }
            if (!shared) {
                usize va = edges[i].a;
                usize vb = edges[i].b;
                const usize vc = p;
                const f64 sa = signed_area2(bwx.data(), bwy.data(), va, vb, vc);
                if (keel::abs(sa) > 1e-15) {
                    if (sa < 0.0) keel::swap(va, vb);
                    next_tris.push_back(Triangle{va, vb, vc});
                }
            }
        }
        keel::swap(cur, next_tris);
    }

    for (const Triangle& t : cur) {
        if (t.v0 < n && t.v1 < n && t.v2 < n) {
            tris.push_back(t);
        }
    }
    return tris;
}

i64 find_bin(ArrayRef<f64> edges, f64 v) noexcept {
    const usize e = edges.size();
    if (e < 2 || v < edges[0] || v > edges[e - 1]) return -1;
    // Binary search for largest i in [0, e-2] with edges[i] <= v.
    usize lo = 0;
    usize hi = e - 2;
    while (lo < hi) {
        const usize mid = lo + (hi - lo + 1) / 2;
        if (edges[mid] <= v) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if (v >= edges[lo] && (v < edges[lo + 1] || (lo == e - 2 && v == edges[lo + 1]))) {
        return as<i64>(lo);
    }
    return -1;
}

} // namespace

// ============================================================================
// 1. 2D Polygons
// ============================================================================

f64 polyarea(ArrayRef<f64> x, ArrayRef<f64> y) noexcept {
    const usize n = keel::min(x.size(), y.size());
    if (n < 3) return 0.0;
    f64 s = 0.0;
    for (usize i = 0; i < n; ++i) {
        const usize j = (i + 1 == n) ? 0 : (i + 1);
        s += x[i] * y[j] - x[j] * y[i];
    }
    return 0.5 * keel::abs(s);
}

bool inpolygon_point(f64 xq, f64 yq, ArrayRef<f64> xv, ArrayRef<f64> yv) noexcept {
    const usize nv = keel::min(xv.size(), yv.size());
    if (nv < 3) return false;
    bool inside = false;
    usize j = nv - 1;
    for (usize i = 0; i < nv; ++i) {
        const f64 xi = xv[i];
        const f64 yi = yv[i];
        const f64 xj = xv[j];
        const f64 yj = yv[j];
        if ((yi > yq) != (yj > yq)) {
            const f64 x_cross = xi + (yq - yi) * (xj - xi) / (yj - yi);
            if (xq < x_cross) inside = !inside;
        }
        j = i;
    }
    return inside;
}

void inpolygon(ArrayRef<f64> xq, ArrayRef<f64> yq, ArrayRef<f64> xv, ArrayRef<f64> yv,
               MutArrayRef<u8> out) noexcept {
    const usize nq = keel::min(keel::min(xq.size(), yq.size()), out.size());
    const usize nv = keel::min(xv.size(), yv.size());
    if (nv < 3) {
        for (usize q = 0; q < nq; ++q) out[q] = 0;
        return;
    }
    for (usize q = 0; q < nq; ++q) {
        out[q] = inpolygon_point(xq[q], yq[q], xv, yv) ? 1 : 0;
    }
}

// ============================================================================
// 2. Convex Hull, Shrink Boundary & 2D Delaunay Triangulation
// ============================================================================

Array<usize> convhull(ArrayRef<f64> x, ArrayRef<f64> y, Allocator* alloc) {
    const usize n = keel::min(x.size(), y.size());
    Array<usize> hull(alloc);
    if (n == 0) return hull;
    if (n < 3) {
        hull.reserve(n + 1);
        for (usize i = 0; i < n; ++i) hull.push_back(i);
        hull.push_back(0);
        return hull;
    }

    ArenaTemp scratch = get_scratch(alloc);
    Array<usize> idx(n, scratch.allocator());
    for (usize i = 0; i < n; ++i) idx[i] = i;
    keel::sort(idx, [&](usize a, usize b) {
        if (x[a] != x[b]) return x[a] < x[b];
        return y[a] < y[b];
    });

    auto cross = [&](usize o, usize a, usize b) noexcept {
        return (x[a] - x[o]) * (y[b] - y[o]) - (y[a] - y[o]) * (x[b] - x[o]);
    };

    hull.reserve(2 * n);
    for (usize i = 0; i < n; ++i) {
        const usize p = idx[i];
        while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    const usize lower_size = hull.size() + 1;
    for (usize i = n - 1; i-- > 0;) {
        const usize p = idx[i];
        while (hull.size() >= lower_size &&
               cross(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    return hull;
}

Array<usize> boundary(ArrayRef<f64> x, ArrayRef<f64> y, f64 shrink, Allocator* alloc) {
    const usize n = keel::min(x.size(), y.size());
    if (!is_finite(shrink)) shrink = 0.0;
    if (shrink < 0.0) shrink = 0.0;
    if (shrink > 1.0) shrink = 1.0;
    if (shrink == 0.0 || n < 4) return convhull(x, y, alloc);

    ArenaTemp scratch = get_scratch(alloc);
    Allocator* salloc = scratch.allocator();
    Array<Triangle> tris = delaunay_impl(x.data(), y.data(), n, salloc);
    if (tris.empty()) return convhull(x, y, alloc);

    auto edge_len = [&](usize a, usize b) noexcept {
        return keel::hypot(x[a] - x[b], y[a] - y[b]);
    };

    Array<f64> longest_edge(tris.size(), salloc);
    for (usize i = 0; i < tris.size(); ++i) {
        const Triangle& t = tris[i];
        longest_edge[i] = keel::max(keel::max(edge_len(t.v0, t.v1), edge_len(t.v1, t.v2)),
                                    edge_len(t.v2, t.v0));
    }

    Array<f64> sorted_len = longest_edge.clone(salloc);
    keel::sort(sorted_len);
    const f64 q = 1.0 - shrink;
    const usize qi =
        keel::min(sorted_len.size() - 1, as<usize>(keel::floor(q * as<f64>(sorted_len.size()))));
    const f64 threshold = sorted_len[qi];

    Array<Triangle> kept(salloc);
    kept.reserve(tris.size());
    for (usize i = 0; i < tris.size(); ++i) {
        if (longest_edge[i] <= threshold) kept.push_back(tris[i]);
    }
    if (kept.empty()) return convhull(x, y, alloc);

    Array<Edge> all_edges(salloc);
    all_edges.reserve(kept.size() * 3);
    auto add_edge = [&](usize a, usize b) {
        if (a > b) keel::swap(a, b);
        all_edges.push_back(Edge{a, b});
    };
    for (const Triangle& t : kept) {
        add_edge(t.v0, t.v1);
        add_edge(t.v1, t.v2);
        add_edge(t.v2, t.v0);
    }
    keel::sort(all_edges);

    Array<SmallArray<AdjEdge, 4>> adj(salloc);
    adj.reserve(n);
    for (usize v = 0; v < n; ++v) adj.emplace_back(salloc);

    usize num_boundary_edges = 0;
    for (usize i = 0; i < all_edges.size();) {
        usize j = i + 1;
        while (j < all_edges.size() && all_edges[j] == all_edges[i]) ++j;
        if (j - i == 1) {
            const usize eid = num_boundary_edges++;
            const Edge& e = all_edges[i];
            adj[e.a].push_back(AdjEdge{e.b, eid});
            adj[e.b].push_back(AdjEdge{e.a, eid});
        }
        i = j;
    }
    if (num_boundary_edges == 0) return convhull(x, y, alloc);

    usize start = n;
    usize num_active_verts = 0;
    for (usize v = 0; v < n; ++v) {
        if (!adj[v].empty()) {
            if (start == n) start = v;
            ++num_active_verts;
        }
    }

    Array<usize> poly(alloc);
    Array<u8> visited_edge(num_boundary_edges, u8{0}, salloc);
    usize cur = start;
    poly.push_back(cur);
    for (usize step = 0; step < num_active_verts + 2; ++step) {
        bool moved = false;
        for (const AdjEdge& ae : adj[cur]) {
            if (visited_edge[ae.id]) continue;
            visited_edge[ae.id] = 1;
            poly.push_back(ae.to);
            cur = ae.to;
            moved = true;
            if (cur == start) {
                moved = false;
                break;
            }
            break;
        }
        if (!moved) break;
    }
    return poly;
}

Array<Triangle> delaunay(ArrayRef<f64> x, ArrayRef<f64> y, Allocator* alloc) {
    const usize n = keel::min(x.size(), y.size());
    return delaunay_impl(x.data(), y.data(), n, alloc);
}

// ============================================================================
// 3. Scattered Data Interpolation
// ============================================================================

bool griddata(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> v, ArrayRef<f64> xq, ArrayRef<f64> yq,
              MutArrayRef<f64> out, ScatteredMethod method, Allocator* scratch_alloc) {
    const usize n_in = x.size();
    if (y.size() != n_in || v.size() != n_in) return false;
    const usize nq = xq.size();
    if (yq.size() != nq || out.size() < nq) return false;

    const f64 nan = Limits<f64>::quiet_nan();
    if (n_in == 0) {
        for (usize q = 0; q < nq; ++q) out[q] = nan;
        return true;
    }

    ArenaTemp scratch = get_scratch(scratch_alloc);
    Allocator* salloc = scratch.allocator();

    // Merge coincident sample points by averaging v.
    Array<f64> ux(n_in, salloc);
    Array<f64> uy(n_in, salloc);
    Array<f64> uv(n_in, salloc);
    Array<usize> counts(n_in, usize{0}, salloc);
    usize n = 0;
    for (usize i = 0; i < n_in; ++i) {
        bool merged = false;
        for (usize j = 0; j < n; ++j) {
            if (keel::abs(ux[j] - x[i]) <= 1e-12 && keel::abs(uy[j] - y[i]) <= 1e-12) {
                uv[j] += v[i];
                counts[j] += 1;
                merged = true;
                break;
            }
        }
        if (!merged) {
            ux[n] = x[i];
            uy[n] = y[i];
            uv[n] = v[i];
            counts[n] = 1;
            ++n;
        }
    }
    for (usize j = 0; j < n; ++j) {
        if (counts[j] > 1) uv[j] /= as<f64>(counts[j]);
    }

    if (method == ScatteredMethod::Nearest) {
        for (usize q = 0; q < nq; ++q) {
            f64 best_d2 = Limits<f64>::infinity();
            usize best_idx = 0;
            for (usize i = 0; i < n; ++i) {
                const f64 dx = xq[q] - ux[i];
                const f64 dy = yq[q] - uy[i];
                const f64 d2 = dx * dx + dy * dy;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_idx = i;
                }
            }
            out[q] = uv[best_idx];
        }
        return true;
    }

    if (method == ScatteredMethod::BiharmonicV4) {
        // Sandwell (1987) biharmonic spline interpolation: G * w = uv.
        Array<f64> g(n * n, 0.0, salloc);
        Array<f64> w(n, 0.0, salloc);
        for (usize i = 0; i < n; ++i) {
            w[i] = uv[i];
            for (usize j = 0; j < n; ++j) {
                if (i == j) continue;
                const f64 d = keel::hypot(ux[i] - ux[j], uy[i] - uy[j]);
                if (d > 0.0) g[i * n + j] = d * d * (keel::log(d) - 1.0);
            }
        }
        // Gaussian elimination with partial pivoting.
        for (usize k = 0; k < n; ++k) {
            usize piv = k;
            f64 max_abs = keel::abs(g[k * n + k]);
            for (usize i = k + 1; i < n; ++i) {
                const f64 a = keel::abs(g[i * n + k]);
                if (a > max_abs) {
                    max_abs = a;
                    piv = i;
                }
            }
            if (piv != k) {
                for (usize j = k; j < n; ++j) keel::swap(g[k * n + j], g[piv * n + j]);
                keel::swap(w[k], w[piv]);
            }
            const f64 diag = g[k * n + k];
            if (keel::abs(diag) < 1e-18) continue;
            for (usize i = k + 1; i < n; ++i) {
                const f64 f = g[i * n + k] / diag;
                g[i * n + k] = 0.0;
                for (usize j = k + 1; j < n; ++j) g[i * n + j] -= f * g[k * n + j];
                w[i] -= f * w[k];
            }
        }
        for (usize k = n; k-- > 0;) {
            f64 s = w[k];
            for (usize j = k + 1; j < n; ++j) s -= g[k * n + j] * w[j];
            const f64 diag = g[k * n + k];
            w[k] = (keel::abs(diag) >= 1e-18) ? (s / diag) : 0.0;
        }
        for (usize q = 0; q < nq; ++q) {
            f64 acc = 0.0;
            for (usize i = 0; i < n; ++i) {
                const f64 d = keel::hypot(xq[q] - ux[i], yq[q] - uy[i]);
                if (d > 0.0) acc += w[i] * d * d * (keel::log(d) - 1.0);
            }
            out[q] = acc;
        }
        return true;
    }

    if (n < 3) {
        for (usize q = 0; q < nq; ++q) out[q] = nan;
        return true;
    }

    Array<Triangle> tris = delaunay_impl(ux.data(), uy.data(), n, salloc);
    for (usize q = 0; q < nq; ++q) {
        const f64 qx = xq[q];
        const f64 qy = yq[q];
        bool found = false;
        for (const Triangle& t : tris) {
            const f64 xa = ux[t.v0];
            const f64 ya = uy[t.v0];
            const f64 xb = ux[t.v1];
            const f64 yb = uy[t.v1];
            const f64 xc = ux[t.v2];
            const f64 yc = uy[t.v2];
            const f64 denom = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc);
            if (keel::abs(denom) < 1e-15) continue;
            const f64 l1 = ((yb - yc) * (qx - xc) + (xc - xb) * (qy - yc)) / denom;
            const f64 l2 = ((yc - ya) * (qx - xc) + (xa - xc) * (qy - yc)) / denom;
            const f64 l3 = 1.0 - l1 - l2;
            if (l1 >= -1e-9 && l2 >= -1e-9 && l3 >= -1e-9) {
                out[q] = l1 * uv[t.v0] + l2 * uv[t.v1] + l3 * uv[t.v2];
                found = true;
                break;
            }
        }
        if (!found) out[q] = nan;
    }
    return true;
}

bool griddatan_nearest(ArrayRef<f64> x_colmajor, usize m, usize dims, ArrayRef<f64> v,
                       ArrayRef<f64> xi_colmajor, usize k, MutArrayRef<f64> out) noexcept {
    if (x_colmajor.size() < m * dims || v.size() < m || xi_colmajor.size() < k * dims ||
        out.size() < k) {
        return false;
    }
    if (m == 0) {
        const f64 nan = Limits<f64>::quiet_nan();
        for (usize q = 0; q < k; ++q) out[q] = nan;
        return true;
    }
    for (usize q = 0; q < k; ++q) {
        f64 best = Limits<f64>::infinity();
        usize best_idx = 0;
        for (usize i = 0; i < m; ++i) {
            f64 d2 = 0.0;
            for (usize c = 0; c < dims; ++c) {
                const f64 dc = xi_colmajor[c * k + q] - x_colmajor[c * m + i];
                d2 += dc * dc;
            }
            if (d2 < best) {
                best = d2;
                best_idx = i;
            }
        }
        out[q] = v[best_idx];
    }
    return true;
}

// ============================================================================
// 4. 2D Histogram Binning
// ============================================================================

void histcounts2(ArrayRef<f64> x, ArrayRef<f64> y, ArrayRef<f64> xedges, ArrayRef<f64> yedges,
                 MutArrayRef<f64> out_colmajor) noexcept {
    if (xedges.size() < 2 || yedges.size() < 2) return;
    const usize nx = xedges.size() - 1;
    const usize ny = yedges.size() - 1;
    if (out_colmajor.size() < nx * ny) return;
    for (usize i = 0; i < nx * ny; ++i) out_colmajor[i] = 0.0;

    const usize n = keel::min(x.size(), y.size());
    for (usize i = 0; i < n; ++i) {
        if (!is_finite(x[i]) || !is_finite(y[i])) continue;
        const i64 bx = find_bin(xedges, x[i]);
        const i64 by = find_bin(yedges, y[i]);
        if (bx < 0 || by < 0) continue;
        out_colmajor[as<usize>(by) * nx + as<usize>(bx)] += 1.0;
    }
}

// ============================================================================
// 5. Linear Assignment / Bipartite Matching (Jonker-Volgenant Hungarian)
// ============================================================================

MatchpairsOutput matchpairs(ArrayRef<f64> cost_colmajor, usize rows, usize cols, f64 cost_unmatched,
                            bool maximize, Allocator* alloc) {
    MatchpairsOutput res(alloc);
    if (rows == 0 || cols == 0 || cost_colmajor.size() < rows * cols) {
        res.unmatched_rows.reserve(rows);
        for (usize i = 0; i < rows; ++i) res.unmatched_rows.push_back(i);
        res.unmatched_cols.reserve(cols);
        for (usize j = 0; j < cols; ++j) res.unmatched_cols.push_back(j);
        return res;
    }

    const usize n = rows + cols;
    constexpr f64 kBig = 1e15;
    ArenaTemp scratch = get_scratch(alloc);
    Allocator* salloc = scratch.allocator();

    Array<f64> a(n * n, kBig, salloc);
    for (usize i = 0; i < rows; ++i) {
        for (usize j = 0; j < cols; ++j) {
            const f64 v = cost_colmajor[j * rows + i];
            a[i * n + j] = maximize ? -v : v;
        }
    }
    const f64 cu_signed = maximize ? -cost_unmatched : cost_unmatched;
    for (usize i = 0; i < rows; ++i) a[i * n + (cols + i)] = cu_signed;
    for (usize j = 0; j < cols; ++j) a[(rows + j) * n + j] = cu_signed;
    for (usize j = 0; j < cols; ++j) {
        for (usize i = 0; i < rows; ++i) a[(rows + j) * n + (cols + i)] = 0.0;
    }

    Array<f64> u(n + 1, 0.0, salloc);
    Array<f64> v_d(n + 1, 0.0, salloc);
    Array<i32> p(n + 1, i32{0}, salloc);
    Array<i32> way(n + 1, i32{0}, salloc);
    Array<f64> minv(salloc);
    Array<u8> used(salloc);

    for (usize i = 1; i <= n; ++i) {
        p[0] = as<i32>(i);
        i32 j0 = 0;
        minv.assign(n + 1, Limits<f64>::infinity());
        used.assign(n + 1, u8{0});
        do {
            used[as<usize>(j0)] = 1;
            const i32 i0 = p[as<usize>(j0)];
            i32 j1 = -1;
            f64 delta = Limits<f64>::infinity();
            for (usize j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const f64 cur = a[as<usize>(i0 - 1) * n + (j - 1)] - u[as<usize>(i0)] - v_d[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = as<i32>(j);
                }
            }
            for (usize j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[as<usize>(p[j])] += delta;
                    v_d[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[as<usize>(j0)] != 0);
        do {
            const i32 j1 = way[as<usize>(j0)];
            p[as<usize>(j0)] = p[as<usize>(j1)];
            j0 = j1;
        } while (j0 != 0);
    }

    Array<i32> row_assign(n, i32{-1}, salloc);
    for (usize j = 1; j <= n; ++j) {
        if (p[j] > 0) row_assign[as<usize>(p[j] - 1)] = as<i32>(j) - 1;
    }

    res.matches.reserve(keel::min(rows, cols));
    res.unmatched_rows.reserve(rows);
    for (usize i = 0; i < rows; ++i) {
        const i32 j = row_assign[i];
        if (j >= 0 && as<usize>(j) < cols) {
            res.matches.push_back(PairIndex{i, as<usize>(j)});
        } else {
            res.unmatched_rows.push_back(i);
        }
    }

    Array<u8> col_used(cols, u8{0}, salloc);
    for (const PairIndex& pr : res.matches) col_used[pr.col] = 1;
    res.unmatched_cols.reserve(cols);
    for (usize j = 0; j < cols; ++j) {
        if (!col_used[j]) res.unmatched_cols.push_back(j);
    }
    return res;
}

} // namespace keel::geom
