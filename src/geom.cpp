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

bool circumcenter(f64 ax, f64 ay, f64 bx, f64 by, f64 cx, f64 cy, f64& ox, f64& oy) noexcept {
    const f64 d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    const f64 scale = max(1.0, max(abs(ax - bx), abs(ay - by)) * max(abs(ax - cx), abs(ay - cy)));
    if (abs(d) <= 1e-14 * scale) return false;
    const f64 a2 = ax * ax + ay * ay;
    const f64 b2 = bx * bx + by * by;
    const f64 c2 = cx * cx + cy * cy;
    ox = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    oy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    return is_finite(ox) && is_finite(oy);
}

Array<Triangle> delaunay_impl(const f64* x, const f64* y, usize n, Allocator* alloc) {
    Array<Triangle> tris(alloc);
    if (n < 3) return tris;

    // Normalize coordinates to [-0.5, 0.5] so circumcircle & area tolerances are scale-invariant,
    // and use Bowyer-Watson for all n >= 3 (avoids co-circular overlapping triangles).
    ArenaTemp scratch = get_scratch(alloc);
    Allocator* salloc = scratch.allocator();

    f64 xmin = Limits<f64>::infinity();
    f64 xmax = -Limits<f64>::infinity();
    f64 ymin = Limits<f64>::infinity();
    f64 ymax = -Limits<f64>::infinity();
    usize valid_count = 0;
    for (usize i = 0; i < n; ++i) {
        if (!is_finite(x[i]) || !is_finite(y[i])) continue;
        if (x[i] < xmin) xmin = x[i];
        if (x[i] > xmax) xmax = x[i];
        if (y[i] < ymin) ymin = y[i];
        if (y[i] > ymax) ymax = y[i];
        ++valid_count;
    }
    if (valid_count < 3) return tris;

    const f64 dmax = keel::max(xmax - xmin, ymax - ymin);
    if (dmax <= 0.0 || !is_finite(dmax)) return tris;
    const f64 inv_d = 1.0 / dmax;
    const f64 xmid = 0.5 * (xmin + xmax);
    const f64 ymid = 0.5 * (ymin + ymax);

    Array<f64> bwx(n + 3, 0.0, salloc);
    Array<f64> bwy(n + 3, 0.0, salloc);
    Array<u8> active(n, u8{0}, salloc);
    for (usize i = 0; i < n; ++i) {
        if (!is_finite(x[i]) || !is_finite(y[i])) continue;
        const f64 nx = (x[i] - xmid) * inv_d;
        const f64 ny = (y[i] - ymid) * inv_d;
        bool dup = false;
        for (usize j = 0; j < i; ++j) {
            if (active[j] && keel::abs(bwx[j] - nx) <= 1e-13 && keel::abs(bwy[j] - ny) <= 1e-13) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            bwx[i] = nx;
            bwy[i] = ny;
            active[i] = 1;
        }
    }

    bwx[n] = -1000.0;
    bwy[n] = -1000.0;
    bwx[n + 1] = 1000.0;
    bwy[n + 1] = -1000.0;
    bwx[n + 2] = 0.0;
    bwy[n + 2] = 1000.0;

    Array<Triangle> cur(salloc);
    Array<Triangle> next_tris(salloc);
    Array<Edge> edges(salloc);
    cur.reserve(2 * n + 4);
    next_tris.reserve(2 * n + 4);
    cur.push_back(Triangle{n, n + 1, n + 2});

    for (usize p = 0; p < n; ++p) {
        if (!active[p]) continue;
        edges.clear();
        next_tris.clear();
        for (const Triangle& t : cur) {
            if (in_circle_det(bwx.data(), bwy.data(), t.v0, t.v1, t.v2, p) > 1e-13) {
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
                if (keel::abs(sa) > 1e-14) {
                    if (sa < 0.0) keel::swap(va, vb);
                    next_tris.push_back(Triangle{va, vb, vc});
                }
            }
        }
        keel::swap(cur, next_tris);
    }

    tris.reserve(cur.size());
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
    if (nv < 3 || !is_finite(xq) || !is_finite(yq)) return false;
    bool inside = false;
    usize j = nv - 1;
    for (usize i = 0; i < nv; ++i) {
        const f64 xi = xv[i];
        const f64 yi = yv[i];
        const f64 xj = xv[j];
        const f64 yj = yv[j];
        // Boundary (edge & vertex) inclusion check.
        const f64 dx = xj - xi;
        const f64 dy = yj - yi;
        const f64 len2 = dx * dx + dy * dy;
        const f64 px = xq - xi;
        const f64 py = yq - yi;
        if (len2 == 0.0) {
            if (px == 0.0 && py == 0.0) return true;
        } else {
            const f64 cross = px * dy - py * dx;
            const f64 dot = px * dx + py * dy;
            if (cross * cross <= 1e-24 * len2 * keel::max(1.0, len2) && dot >= -1e-12 * len2 &&
                dot <= (1.0 + 1e-12) * len2) {
                return true;
            }
        }
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

    ArenaTemp scratch = get_scratch(alloc);
    Array<usize> idx(scratch.allocator());
    idx.reserve(n);
    for (usize i = 0; i < n; ++i) {
        if (is_finite(x[i]) && is_finite(y[i])) idx.push_back(i);
    }
    const usize m = idx.size();
    if (m == 0) return hull;
    if (m < 3) {
        hull.reserve(m + 1);
        for (usize i = 0; i < m; ++i) hull.push_back(idx[i]);
        hull.push_back(idx[0]);
        return hull;
    }

    keel::sort(idx, [&](usize a, usize b) {
        if (x[a] != x[b]) return x[a] < x[b];
        return y[a] < y[b];
    });

    auto cross = [&](usize o, usize a, usize b) noexcept {
        return (x[a] - x[o]) * (y[b] - y[o]) - (y[a] - y[o]) * (x[b] - x[o]);
    };

    hull.reserve(2 * m);
    for (usize i = 0; i < m; ++i) {
        const usize p = idx[i];
        while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }
    const usize lower_size = hull.size() + 1;
    for (usize i = m - 1; i-- > 0;) {
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

    // Merge coincident finite sample points by averaging v.
    Array<f64> ux(n_in, salloc);
    Array<f64> uy(n_in, salloc);
    Array<f64> uv(n_in, salloc);
    Array<usize> counts(n_in, usize{0}, salloc);
    usize n = 0;
    for (usize i = 0; i < n_in; ++i) {
        if (!is_finite(x[i]) || !is_finite(y[i]) || !is_finite(v[i])) continue;
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
    if (n == 0) {
        for (usize q = 0; q < nq; ++q) out[q] = nan;
        return true;
    }
    for (usize j = 0; j < n; ++j) {
        if (counts[j] > 1) uv[j] /= as<f64>(counts[j]);
    }

    if (method == ScatteredMethod::Nearest) {
        for (usize q = 0; q < nq; ++q) {
            if (!is_finite(xq[q]) || !is_finite(yq[q])) {
                out[q] = nan;
                continue;
            }
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
        if (n == 1) {
            for (usize q = 0; q < nq; ++q) {
                out[q] = (is_finite(xq[q]) && is_finite(yq[q])) ? uv[0] : nan;
            }
            return true;
        }
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
            if (!is_finite(xq[q]) || !is_finite(yq[q])) {
                out[q] = nan;
                continue;
            }
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

    if (method == ScatteredMethod::Cubic) {
        // Build 1-ring vertex adjacency from Delaunay triangles.
        Array<usize> deg(n, usize{0}, salloc);
        for (const Triangle& t : tris) {
            deg[t.v0] += 2;
            deg[t.v1] += 2;
            deg[t.v2] += 2;
        }
        Array<usize> adj_off(n + 1, usize{0}, salloc);
        for (usize i = 0; i < n; ++i) adj_off[i + 1] = adj_off[i] + deg[i];
        Array<usize> adj_cur(n, usize{0}, salloc);
        for (usize i = 0; i < n; ++i) adj_cur[i] = adj_off[i];
        Array<usize> adj(adj_off[n], usize{0}, salloc);
        for (const Triangle& t : tris) {
            adj[adj_cur[t.v0]++] = t.v1;
            adj[adj_cur[t.v0]++] = t.v2;
            adj[adj_cur[t.v1]++] = t.v0;
            adj[adj_cur[t.v1]++] = t.v2;
            adj[adj_cur[t.v2]++] = t.v0;
            adj[adj_cur[t.v2]++] = t.v1;
        }

        // Estimate vertex gradients (gx, gy) via weighted local quadratic least-squares
        // over 1-ring + 2-ring Delaunay neighbors (exact for all degree <= 2 polynomials).
        Array<f64> gx(n, 0.0, salloc);
        Array<f64> gy(n, 0.0, salloc);
        Array<u32> stamp(n, u32{0}, salloc);
        Array<usize> nbrs(salloc);
        u32 cur_stamp = 0;

        auto solve_small = [](f64* a, f64* b, usize m) noexcept -> bool {
            for (usize k = 0; k < m; ++k) {
                usize piv = k;
                f64 best = keel::abs(a[k * m + k]);
                for (usize i = k + 1; i < m; ++i) {
                    const f64 v = keel::abs(a[i * m + k]);
                    if (v > best) {
                        best = v;
                        piv = i;
                    }
                }
                if (best < 1e-14) return false;
                if (piv != k) {
                    for (usize j = k; j < m; ++j) keel::swap(a[k * m + j], a[piv * m + j]);
                    keel::swap(b[k], b[piv]);
                }
                const f64 diag = a[k * m + k];
                for (usize i = k + 1; i < m; ++i) {
                    const f64 f = a[i * m + k] / diag;
                    for (usize j = k + 1; j < m; ++j) a[i * m + j] -= f * a[k * m + j];
                    b[i] -= f * b[k];
                }
            }
            for (usize k = m; k-- > 0;) {
                f64 s = b[k];
                for (usize j = k + 1; j < m; ++j) s -= a[k * m + j] * b[j];
                b[k] = s / a[k * m + k];
            }
            return true;
        };

        for (usize i = 0; i < n; ++i) {
            ++cur_stamp;
            stamp[i] = cur_stamp;
            nbrs.clear();
            const usize r1_beg = adj_off[i];
            const usize r1_end = adj_off[i + 1];
            for (usize e = r1_beg; e < r1_end; ++e) {
                const usize u = adj[e];
                if (stamp[u] != cur_stamp) {
                    stamp[u] = cur_stamp;
                    nbrs.push_back(u);
                }
            }
            const usize n1 = nbrs.size();
            if (n1 < 6) {
                for (usize k = 0; k < n1; ++k) {
                    const usize u = nbrs[k];
                    for (usize e = adj_off[u]; e < adj_off[u + 1]; ++e) {
                        const usize w = adj[e];
                        if (stamp[w] != cur_stamp) {
                            stamp[w] = cur_stamp;
                            nbrs.push_back(w);
                        }
                    }
                }
            }

            // Compute local characteristic scale so normal equations stay well-conditioned.
            f64 scale = 0.0;
            for (usize k = 0; k < nbrs.size(); ++k) {
                const usize u = nbrs[k];
                const f64 d = keel::hypot(ux[u] - ux[i], uy[u] - uy[i]);
                if (d > scale) scale = d;
            }
            if (scale <= 0.0) continue;
            const f64 inv_s = 1.0 / scale;

            bool solved = false;
            if (nbrs.size() >= 5) {
                f64 ata[25] = {};
                f64 atb[5] = {};
                for (usize k = 0; k < nbrs.size(); ++k) {
                    const usize u = nbrs[k];
                    const f64 dx = (ux[u] - ux[i]) * inv_s;
                    const f64 dy = (uy[u] - uy[i]) * inv_s;
                    const f64 r2 = dx * dx + dy * dy;
                    if (r2 < 1e-24) continue;
                    const f64 wt = 1.0 / (r2 + 1e-6);
                    const f64 dv = uv[u] - uv[i];
                    const f64 row[5] = {dx, dy, 0.5 * dx * dx, dx * dy, 0.5 * dy * dy};
                    for (usize r = 0; r < 5; ++r) {
                        atb[r] += wt * row[r] * dv;
                        for (usize c = 0; c < 5; ++c) ata[r * 5 + c] += wt * row[r] * row[c];
                    }
                }
                if (solve_small(ata, atb, 5)) {
                    gx[i] = atb[0] * inv_s;
                    gy[i] = atb[1] * inv_s;
                    solved = true;
                }
            }
            if (!solved && nbrs.size() >= 2) {
                f64 ata[4] = {};
                f64 atb[2] = {};
                for (usize k = 0; k < nbrs.size(); ++k) {
                    const usize u = nbrs[k];
                    const f64 dx = (ux[u] - ux[i]) * inv_s;
                    const f64 dy = (uy[u] - uy[i]) * inv_s;
                    const f64 r2 = dx * dx + dy * dy;
                    if (r2 < 1e-24) continue;
                    const f64 wt = 1.0 / (r2 + 1e-6);
                    const f64 dv = uv[u] - uv[i];
                    ata[0] += wt * dx * dx;
                    ata[1] += wt * dx * dy;
                    ata[2] += wt * dy * dx;
                    ata[3] += wt * dy * dy;
                    atb[0] += wt * dx * dv;
                    atb[1] += wt * dy * dv;
                }
                if (solve_small(ata, atb, 2)) {
                    gx[i] = atb[0] * inv_s;
                    gy[i] = atb[1] * inv_s;
                }
            }
        }

        // Clough-Tocher C1 macro-element evaluator on a single triangle.
        auto eval_ct = [](f64 x1, f64 y1, f64 x2, f64 y2, f64 x3, f64 y3, f64 f1, f64 f2, f64 f3,
                          f64 gx1, f64 gy1, f64 gx2, f64 gy2, f64 gx3, f64 gy3, f64 b1, f64 b2,
                          f64 b3) noexcept -> f64 {
            const f64 c3000 = f1;
            const f64 c0300 = f2;
            const f64 c0030 = f3;

            const f64 c2100 = f1 + (gx1 * (x2 - x1) + gy1 * (y2 - y1)) / 3.0;
            const f64 c2010 = f1 + (gx1 * (x3 - x1) + gy1 * (y3 - y1)) / 3.0;
            const f64 c1200 = f2 + (gx2 * (x1 - x2) + gy2 * (y1 - y2)) / 3.0;
            const f64 c0210 = f2 + (gx2 * (x3 - x2) + gy2 * (y3 - y2)) / 3.0;
            const f64 c1020 = f3 + (gx3 * (x1 - x3) + gy3 * (y1 - y3)) / 3.0;
            const f64 c0120 = f3 + (gx3 * (x2 - x3) + gy3 * (y2 - y3)) / 3.0;

            const f64 c2001 = (c3000 + c2100 + c2010) / 3.0;
            const f64 c0201 = (c0300 + c1200 + c0210) / 3.0;
            const f64 c0021 = (c0030 + c1020 + c0120) / 3.0;

            const f64 x0 = (x1 + x2 + x3) / 3.0;
            const f64 y0 = (y1 + y2 + y3) / 3.0;

            auto mid_face = [&](f64 xj, f64 yj, f64 xk, f64 yk, f64 cjjj, f64 cjjk, f64 cjkk,
                                f64 ckkk, f64 cjj0, f64 ckk0) noexcept -> f64 {
                const f64 ex = xk - xj;
                const f64 ey = yk - yj;
                const f64 e2 = ex * ex + ey * ey;
                const f64 lam = (e2 > 0.0) ? (((x0 - xj) * ex + (y0 - yj) * ey) / e2) : 0.5;
                const f64 d0 = (cjj0 - cjjj) - lam * (cjjk - cjjj);
                const f64 d2 = (ckk0 - cjkk) - lam * (ckkk - cjkk);
                return cjjk + lam * (cjkk - cjjk) + 0.5 * (d0 + d2);
            };

            const f64 c0111 = mid_face(x2, y2, x3, y3, c0300, c0210, c0120, c0030, c0201, c0021);
            const f64 c1011 = mid_face(x3, y3, x1, y1, c0030, c1020, c2010, c3000, c0021, c2001);
            const f64 c1101 = mid_face(x1, y1, x2, y2, c3000, c2100, c1200, c0300, c2001, c0201);

            const f64 c1002 = (c2001 + c1101 + c1011) / 3.0;
            const f64 c0102 = (c0201 + c1101 + c0111) / 3.0;
            const f64 c0012 = (c0021 + c1011 + c0111) / 3.0;
            const f64 c0003 = (c1002 + c0102 + c0012) / 3.0;

            auto eval_bernstein = [](f64 u, f64 v, f64 w, f64 c300, f64 c030, f64 c003, f64 c210,
                                     f64 c120, f64 c201, f64 c021, f64 c102, f64 c012,
                                     f64 c111) noexcept -> f64 {
                const f64 u2 = u * u;
                const f64 v2 = v * v;
                const f64 w2 = w * w;
                return c300 * u2 * u + c030 * v2 * v + c003 * w2 * w +
                       3.0 * (c210 * u2 * v + c120 * u * v2 + c201 * u2 * w + c021 * v2 * w +
                              c102 * u * w2 + c012 * v * w2) +
                       6.0 * c111 * u * v * w;
            };

            if (b1 <= b2 && b1 <= b3) {
                return eval_bernstein(b2 - b1, b3 - b1, 3.0 * b1, c0300, c0030, c0003, c0210, c0120,
                                      c0201, c0021, c0102, c0012, c0111);
            }
            if (b2 <= b1 && b2 <= b3) {
                return eval_bernstein(b3 - b2, b1 - b2, 3.0 * b2, c0030, c3000, c0003, c1020, c2010,
                                      c0021, c2001, c0012, c1002, c1011);
            }
            return eval_bernstein(b1 - b3, b2 - b3, 3.0 * b3, c3000, c0300, c0003, c2100, c1200,
                                  c2001, c0201, c1002, c0102, c1101);
        };

        for (usize q = 0; q < nq; ++q) {
            const f64 qx = xq[q];
            const f64 qy = yq[q];
            if (!is_finite(qx) || !is_finite(qy)) {
                out[q] = nan;
                continue;
            }
            bool found = false;
            for (const Triangle& t : tris) {
                const f64 xa = ux[t.v0];
                const f64 ya = uy[t.v0];
                const f64 xb = ux[t.v1];
                const f64 yb = uy[t.v1];
                const f64 xc = ux[t.v2];
                const f64 yc = uy[t.v2];
                const f64 denom = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc);
                if (denom == 0.0) continue;
                const f64 l1 = ((yb - yc) * (qx - xc) + (xc - xb) * (qy - yc)) / denom;
                const f64 l2 = ((yc - ya) * (qx - xc) + (xa - xc) * (qy - yc)) / denom;
                const f64 l3 = 1.0 - l1 - l2;
                if (l1 >= -1e-9 && l2 >= -1e-9 && l3 >= -1e-9) {
                    out[q] = eval_ct(xa, ya, xb, yb, xc, yc, uv[t.v0], uv[t.v1], uv[t.v2],
                                     gx[t.v0], gy[t.v0], gx[t.v1], gy[t.v1], gx[t.v2], gy[t.v2],
                                     l1, l2, l3);
                    found = true;
                    break;
                }
            }
            if (!found) out[q] = nan;
        }
        return true;
    }

    if (method == ScatteredMethod::Natural) {
        // Sibson natural-neighbor interpolation via Watson's circumcenter area decomposition.
        Array<f64> tcx(tris.size(), 0.0, salloc);
        Array<f64> tcy(tris.size(), 0.0, salloc);
        Array<f64> tr2(tris.size(), 0.0, salloc);
        Array<u8> tvalid(tris.size(), u8{0}, salloc);
        for (usize ti = 0; ti < tris.size(); ++ti) {
            const Triangle& t = tris[ti];
            f64 ox = 0.0;
            f64 oy = 0.0;
            if (circumcenter(ux[t.v0], uy[t.v0], ux[t.v1], uy[t.v1], ux[t.v2], uy[t.v2], ox, oy)) {
                tcx[ti] = ox;
                tcy[ti] = oy;
                const f64 dx = ux[t.v0] - ox;
                const f64 dy = uy[t.v0] - oy;
                tr2[ti] = dx * dx + dy * dy;
                tvalid[ti] = 1;
            }
        }

        auto tri_area2 = [](f64 ax, f64 ay, f64 bx, f64 by, f64 cx, f64 cy) noexcept -> f64 {
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        };

        Array<f64> weights(n, 0.0, salloc);
        Array<usize> touched(salloc);
        Array<u32> stamp(n, u32{0}, salloc);
        u32 cur_stamp = 0;

        for (usize q = 0; q < nq; ++q) {
            const f64 qx0 = xq[q];
            const f64 qy0 = yq[q];
            if (!is_finite(qx0) || !is_finite(qy0)) {
                out[q] = nan;
                continue;
            }

            // Locate containing triangle (ensures query is inside convex hull).
            usize hit_tri = kInvalidIndex;
            f64 hit_l1 = 0.0;
            f64 hit_l2 = 0.0;
            f64 hit_l3 = 0.0;
            for (usize ti = 0; ti < tris.size(); ++ti) {
                const Triangle& t = tris[ti];
                const f64 xa = ux[t.v0];
                const f64 ya = uy[t.v0];
                const f64 xb = ux[t.v1];
                const f64 yb = uy[t.v1];
                const f64 xc = ux[t.v2];
                const f64 yc = uy[t.v2];
                const f64 denom = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc);
                if (denom == 0.0) continue;
                const f64 l1 = ((yb - yc) * (qx0 - xc) + (xc - xb) * (qy0 - yc)) / denom;
                const f64 l2 = ((yc - ya) * (qx0 - xc) + (xa - xc) * (qy0 - yc)) / denom;
                const f64 l3 = 1.0 - l1 - l2;
                if (l1 >= -1e-9 && l2 >= -1e-9 && l3 >= -1e-9) {
                    hit_tri = ti;
                    hit_l1 = l1;
                    hit_l2 = l2;
                    hit_l3 = l3;
                    break;
                }
            }
            if (hit_tri == kInvalidIndex) {
                out[q] = nan;
                continue;
            }

            const Triangle& ht = tris[hit_tri];
            // Exact vertex match check.
            if (hit_l1 > 1.0 - 1e-12) {
                out[q] = uv[ht.v0];
                continue;
            }
            if (hit_l2 > 1.0 - 1e-12) {
                out[q] = uv[ht.v1];
                continue;
            }
            if (hit_l3 > 1.0 - 1e-12) {
                out[q] = uv[ht.v2];
                continue;
            }

            // Evaluate Watson's stolen-area decomposition; nudge slightly inward if on a Delaunay edge.
            auto eval_watson = [&](f64 qx, f64 qy, f64& val_out) -> bool {
                ++cur_stamp;
                touched.clear();
                auto add_w = [&](usize v_idx, f64 area) {
                    if (stamp[v_idx] != cur_stamp) {
                        stamp[v_idx] = cur_stamp;
                        weights[v_idx] = 0.0;
                        touched.push_back(v_idx);
                    }
                    weights[v_idx] += area;
                };
                for (usize ti = 0; ti < tris.size(); ++ti) {
                    if (!tvalid[ti]) continue;
                    const f64 dx = qx - tcx[ti];
                    const f64 dy = qy - tcy[ti];
                    if (dx * dx + dy * dy > tr2[ti] * (1.0 + 1e-11)) continue;
                    const Triangle& t = tris[ti];
                    f64 c0x = 0.0, c0y = 0.0;
                    f64 c1x = 0.0, c1y = 0.0;
                    f64 c2x = 0.0, c2y = 0.0;
                    if (!circumcenter(qx, qy, ux[t.v1], uy[t.v1], ux[t.v2], uy[t.v2], c0x, c0y) ||
                        !circumcenter(ux[t.v0], uy[t.v0], qx, qy, ux[t.v2], uy[t.v2], c1x, c1y) ||
                        !circumcenter(ux[t.v0], uy[t.v0], ux[t.v1], uy[t.v1], qx, qy, c2x, c2y)) {
                        return false;
                    }
                    add_w(t.v0, tri_area2(tcx[ti], tcy[ti], c1x, c1y, c2x, c2y));
                    add_w(t.v1, tri_area2(tcx[ti], tcy[ti], c2x, c2y, c0x, c0y));
                    add_w(t.v2, tri_area2(tcx[ti], tcy[ti], c0x, c0y, c1x, c1y));
                }
                f64 wsum = 0.0;
                f64 vsum = 0.0;
                for (usize idx : touched) {
                    wsum += weights[idx];
                    vsum += weights[idx] * uv[idx];
                }
                if (wsum <= 1e-30 || !is_finite(wsum)) return false;
                val_out = vsum / wsum;
                return is_finite(val_out);
            };

            f64 nat_val = 0.0;
            if (!eval_watson(qx0, qy0, nat_val)) {
                const f64 cx_tri = (ux[ht.v0] + ux[ht.v1] + ux[ht.v2]) / 3.0;
                const f64 cy_tri = (uy[ht.v0] + uy[ht.v1] + uy[ht.v2]) / 3.0;
                const f64 qx_n = qx0 + 1e-9 * (cx_tri - qx0);
                const f64 qy_n = qy0 + 1e-9 * (cy_tri - qy0);
                if (!eval_watson(qx_n, qy_n, nat_val)) {
                    nat_val = hit_l1 * uv[ht.v0] + hit_l2 * uv[ht.v1] + hit_l3 * uv[ht.v2];
                }
            }
            out[q] = nat_val;
        }
        return true;
    }

    for (usize q = 0; q < nq; ++q) {
        const f64 qx = xq[q];
        const f64 qy = yq[q];
        if (!is_finite(qx) || !is_finite(qy)) {
            out[q] = nan;
            continue;
        }
        bool found = false;
        for (const Triangle& t : tris) {
            const f64 xa = ux[t.v0];
            const f64 ya = uy[t.v0];
            const f64 xb = ux[t.v1];
            const f64 yb = uy[t.v1];
            const f64 xc = ux[t.v2];
            const f64 yc = uy[t.v2];
            const f64 denom = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc);
            if (denom == 0.0) continue;
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
    if (dims == 0 || (m > 0 && x_colmajor.size() / dims < m) || v.size() < m ||
        (k > 0 && xi_colmajor.size() / dims < k) || out.size() < k) {
        return false;
    }
    const f64 nan = Limits<f64>::quiet_nan();
    if (m == 0) {
        for (usize q = 0; q < k; ++q) out[q] = nan;
        return true;
    }
    for (usize q = 0; q < k; ++q) {
        bool q_finite = true;
        for (usize c = 0; c < dims; ++c) {
            if (!is_finite(xi_colmajor[c * k + q])) {
                q_finite = false;
                break;
            }
        }
        if (!q_finite) {
            out[q] = nan;
            continue;
        }
        f64 best = Limits<f64>::infinity();
        usize best_idx = kInvalidIndex;
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
        out[q] = (best_idx != kInvalidIndex) ? v[best_idx] : nan;
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
    if (rows == 0 || cols == 0 || cost_colmajor.size() / cols < rows || is_nan(cost_unmatched)) {
        res.unmatched_rows.reserve(rows);
        for (usize i = 0; i < rows; ++i) res.unmatched_rows.push_back(i);
        res.unmatched_cols.reserve(cols);
        for (usize j = 0; j < cols; ++j) res.unmatched_cols.push_back(j);
        return res;
    }

    const usize n = rows + cols;
    f64 max_finite = is_finite(cost_unmatched) ? keel::abs(cost_unmatched) : 0.0;
    for (usize i = 0; i < rows; ++i) {
        for (usize j = 0; j < cols; ++j) {
            const f64 v = cost_colmajor[j * rows + i];
            if (is_finite(v)) max_finite = keel::max(max_finite, keel::abs(v));
        }
    }
    // Scale forbidden-arc sentinel dynamically above any finite total assignment cost.
    const f64 big_cost = keel::max(1e15, 1e6 * (as<f64>(n) + 1.0) * (max_finite + 1.0));
    const f64 cu_raw = maximize ? -cost_unmatched : cost_unmatched;
    const f64 cu_signed = is_finite(cu_raw) ? cu_raw : ((cu_raw > 0.0) ? (0.25 * big_cost) : (-0.25 * big_cost));

    ArenaTemp scratch = get_scratch(alloc);
    Allocator* salloc = scratch.allocator();

    Array<f64> a(n * n, big_cost, salloc);
    for (usize i = 0; i < rows; ++i) {
        for (usize j = 0; j < cols; ++j) {
            const f64 v = cost_colmajor[j * rows + i];
            if (is_nan(v)) {
                a[i * n + j] = big_cost;
            } else {
                const f64 cv = maximize ? -v : v;
                a[i * n + j] = is_finite(cv) ? cv : ((cv > 0.0) ? big_cost : (-0.25 * big_cost));
            }
        }
    }
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
            if (j1 < 0) break;
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
        if (j0 < 0) continue;
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
            const f64 orig = cost_colmajor[as<usize>(j) * rows + i];
            const f64 cv = maximize ? -orig : orig;
            if (!is_nan(orig) && (!is_inf(cv) || cv < 0.0)) {
                res.matches.push_back(PairIndex{i, as<usize>(j)});
            } else {
                res.unmatched_rows.push_back(i);
            }
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
