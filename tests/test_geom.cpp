#include "keel/geom.h"
#include "keel/format.h"
#include "keel/math.h"
#include "keel/string.h"

namespace {

keel::i32 g_failed = 0;

void check(bool cond, const char* expr, keel::i32 line) {
    if (!cond) {
        keel::eprintln("FAIL [line %d]: %s", line, expr);
        ++g_failed;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* expr, keel::i32 line) {
    if (keel::is_nan(got) || keel::abs(got - expected) > tol) {
        keel::eprintln("FAIL [line %d]: %s (got %.15g, expected %.15g)", line, expr, got, expected);
        ++g_failed;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_NEAR(got, expected, tol) check_near((got), (expected), (tol), #got " ~= " #expected, __LINE__)

void test_polygon_and_hull() {
    const keel::f64 sq_x[4] = {0.0, 2.0, 2.0, 0.0};
    const keel::f64 sq_y[4] = {0.0, 0.0, 3.0, 3.0};
    CHECK_NEAR(keel::geom::polyarea({sq_x, 4}, {sq_y, 4}), 6.0, 1e-14);
    CHECK(keel::geom::inpolygon_point(1.0, 1.5, {sq_x, 4}, {sq_y, 4}));
    CHECK(!keel::geom::inpolygon_point(3.0, 1.5, {sq_x, 4}, {sq_y, 4}));
    // Boundary edges and vertices must be included (matching MATLAB inpolygon)
    CHECK(keel::geom::inpolygon_point(0.0, 0.0, {sq_x, 4}, {sq_y, 4}));
    CHECK(keel::geom::inpolygon_point(2.0, 1.5, {sq_x, 4}, {sq_y, 4}));
    CHECK(keel::geom::inpolygon_point(1.0, 3.0, {sq_x, 4}, {sq_y, 4}));
    CHECK(!keel::geom::inpolygon_point(keel::kNaN, 1.0, {sq_x, 4}, {sq_y, 4}));

    const keel::f64 px[6] = {0.0, 2.0, 2.0, 0.0, 1.0, 0.5};
    const keel::f64 py[6] = {0.0, 0.0, 2.0, 2.0, 1.0, 1.2};
    auto hull = keel::geom::convhull({px, 6}, {py, 6});
    CHECK(hull.size() == 5);
    CHECK(hull[0] == hull[4]);

    // Convhull with NaN point safely ignores NaN
    const keel::f64 px_nan[5] = {0.0, 2.0, keel::kNaN, 2.0, 0.0};
    const keel::f64 py_nan[5] = {0.0, 0.0, 1.0, 2.0, 2.0};
    auto hull_nan = keel::geom::convhull({px_nan, 5}, {py_nan, 5});
    CHECK(hull_nan.size() == 5 && hull_nan[0] == hull_nan[4]);

    auto bnd = keel::geom::boundary({px, 6}, {py, 6}, 0.0);
    CHECK(bnd.size() == 5);
}

void test_delaunay_and_griddata() {
    // Co-circular unit square MUST emit exactly 2 non-overlapping triangles (not 4!)
    const keel::f64 sx[4] = {0.0, 1.0, 1.0, 0.0};
    const keel::f64 sy[4] = {0.0, 0.0, 1.0, 1.0};
    auto sq_tris = keel::geom::delaunay({sx, 4}, {sy, 4});
    CHECK(sq_tris.size() == 2);

    // Tiny scale invariance (1e-7 square + center -> 4 triangles)
    const keel::f64 tx[5] = {0.0, 1e-7, 1e-7, 0.0, 0.5e-7};
    const keel::f64 ty[5] = {0.0, 0.0, 1e-7, 1e-7, 0.5e-7};
    auto tiny_tris = keel::geom::delaunay({tx, 5}, {ty, 5});
    CHECK(tiny_tris.size() == 4);

    // Small cloud: square + center -> 4 triangles.
    const keel::f64 x[5] = {0.0, 1.0, 1.0, 0.0, 0.5};
    const keel::f64 y[5] = {0.0, 0.0, 1.0, 1.0, 0.5};
    auto tris = keel::geom::delaunay({x, 5}, {y, 5});
    CHECK(tris.size() == 4);

    // Larger cloud: 7x7 grid = 49 points -> 72 triangles.
    keel::Array<keel::f64> gx;
    keel::Array<keel::f64> gy;
    keel::Array<keel::f64> gv;
    for (keel::usize i = 0; i < 7; ++i) {
        for (keel::usize j = 0; j < 7; ++j) {
            const keel::f64 xx = keel::as<keel::f64>(i) + 0.01 * keel::as<keel::f64>((j * 3) % 5);
            const keel::f64 yy = keel::as<keel::f64>(j) + 0.01 * keel::as<keel::f64>((i * 2) % 5);
            gx.push_back(xx);
            gy.push_back(yy);
            gv.push_back(2.0 * xx + 3.0 * yy + 1.0);
        }
    }
    auto big_tris = keel::geom::delaunay(gx.as_ref(), gy.as_ref());
    auto big_hull = keel::geom::convhull(gx.as_ref(), gy.as_ref());
    // Exact non-degenerate simplex count verified against scipy.spatial.Delaunay
    // (49 pts, 11 boundary pts incl. 2 collinear hull vertices -> 2*49 - 2 - 11 = 85 tris)
    // and total triangle area must equal convex hull area.
    CHECK(big_tris.size() == 85);
    keel::f64 sum_tri_area = 0.0;
    for (const auto& t : big_tris) {
        const keel::f64 tx3[3] = {gx[t.v0], gx[t.v1], gx[t.v2]};
        const keel::f64 ty3[3] = {gy[t.v0], gy[t.v1], gy[t.v2]};
        sum_tri_area += keel::geom::polyarea({tx3, 3}, {ty3, 3});
    }
    keel::Array<keel::f64> bhx;
    keel::Array<keel::f64> bhy;
    for (keel::usize i = 0; i + 1 < big_hull.size(); ++i) {
        bhx.push_back(gx[big_hull[i]]);
        bhy.push_back(gy[big_hull[i]]);
    }
    CHECK_NEAR(sum_tri_area, keel::geom::polyarea(bhx.as_ref(), bhy.as_ref()), 1e-11);

    // Linear & Natural reproduce affine functions exactly inside hull, NaN outside.
    const keel::f64 qx[3] = {2.5, 100.0, keel::kNaN};
    const keel::f64 qy[3] = {3.5, 100.0, 1.0};
    keel::f64 qout[3] = {0.0, 0.0, 0.0};
    CHECK(keel::geom::griddata(gx.as_ref(), gy.as_ref(), gv.as_ref(), {qx, 3}, {qy, 3}, {qout, 3},
                               keel::geom::ScatteredMethod::Linear));
    CHECK_NEAR(qout[0], 2.0 * 2.5 + 3.0 * 3.5 + 1.0, 1e-12);
    CHECK(keel::is_nan(qout[1]));
    CHECK(keel::is_nan(qout[2]));

    CHECK(keel::geom::griddata(gx.as_ref(), gy.as_ref(), gv.as_ref(), {qx, 3}, {qy, 3}, {qout, 3},
                               keel::geom::ScatteredMethod::Natural));
    CHECK_NEAR(qout[0], 2.0 * 2.5 + 3.0 * 3.5 + 1.0, 1e-11);
    CHECK(keel::is_nan(qout[1]));
    CHECK(keel::is_nan(qout[2]));

    // Cubic Clough-Tocher reproduces quadratic polynomials to high precision inside hull.
    keel::Array<keel::f64> qv;
    for (keel::usize i = 0; i < gx.size(); ++i) {
        const keel::f64 xx = gx[i];
        const keel::f64 yy = gy[i];
        qv.push_back(1.5 * xx * xx - 0.7 * xx * yy + 2.2 * yy * yy - 3.0 * xx + 1.0);
    }
    keel::f64 qcout[3] = {0.0, 0.0, 0.0};
    CHECK(keel::geom::griddata(gx.as_ref(), gy.as_ref(), qv.as_ref(), {qx, 3}, {qy, 3}, {qcout, 3},
                               keel::geom::ScatteredMethod::Cubic));
    const keel::f64 expected_q = 1.5 * 2.5 * 2.5 - 0.7 * 2.5 * 3.5 + 2.2 * 3.5 * 3.5 - 3.0 * 2.5 + 1.0;
    CHECK_NEAR(qcout[0], expected_q, 1e-10);
    CHECK(keel::is_nan(qcout[1]));
    CHECK(keel::is_nan(qcout[2]));

    // Nearest & BiharmonicV4 including NaN query guard and single-point V4.
    const keel::f64 v5[5] = {0.0, 1.0, 2.0, 1.0, 1.0};
    const keel::f64 q1x[2] = {0.5, keel::kNaN};
    const keel::f64 q1y[2] = {0.5, 0.5};
    keel::f64 q1out[2] = {0.0, 0.0};
    CHECK(keel::geom::griddata({x, 5}, {y, 5}, {v5, 5}, {q1x, 2}, {q1y, 2}, {q1out, 2},
                               keel::geom::ScatteredMethod::Nearest));
    CHECK_NEAR(q1out[0], 1.0, 1e-14);
    CHECK(keel::is_nan(q1out[1]));
    CHECK(keel::geom::griddata({x, 5}, {y, 5}, {v5, 5}, {q1x, 2}, {q1y, 2}, {q1out, 2},
                               keel::geom::ScatteredMethod::BiharmonicV4));
    CHECK_NEAR(q1out[0], 1.0, 1e-10);
    CHECK(keel::is_nan(q1out[1]));

    const keel::f64 x1[1] = {1.0};
    const keel::f64 y1[1] = {2.0};
    const keel::f64 v1[1] = {7.5};
    CHECK(keel::geom::griddata({x1, 1}, {y1, 1}, {v1, 1}, {q1x, 2}, {q1y, 2}, {q1out, 2},
                               keel::geom::ScatteredMethod::BiharmonicV4));
    CHECK_NEAR(q1out[0], 7.5, 1e-14);
    CHECK(keel::is_nan(q1out[1]));

    // ND nearest including NaN query.
    const keel::f64 nd_x[6] = {0.0, 10.0, 0.0, 10.0, 0.0, 10.0};
    const keel::f64 nd_v[2] = {42.0, 99.0};
    const keel::f64 nd_q[6] = {9.0, keel::kNaN, 9.0, 0.0, 9.0, 0.0}; // 2x3 col-major
    keel::f64 nd_out[2] = {0.0, 0.0};
    CHECK(keel::geom::griddatan_nearest({nd_x, 6}, 2, 3, {nd_v, 2}, {nd_q, 6}, 2, {nd_out, 2}));
    CHECK_NEAR(nd_out[0], 99.0, 1e-14);
    CHECK(keel::is_nan(nd_out[1]));
}

void test_histcounts2_and_matchpairs() {
    const keel::f64 hx[4] = {0.5, 1.5, 2.0, 10.0};
    const keel::f64 hy[4] = {0.5, 0.5, 2.0, 0.5};
    const keel::f64 xe[3] = {0.0, 1.0, 2.0};
    const keel::f64 ye[3] = {0.0, 1.0, 2.0};
    keel::f64 counts[4] = {0.0, 0.0, 0.0, 0.0};
    keel::geom::histcounts2({hx, 4}, {hy, 4}, {xe, 3}, {ye, 3}, {counts, 4});
    CHECK(counts[0] == 1.0);
    CHECK(counts[1] == 1.0);
    CHECK(counts[2] == 0.0);
    CHECK(counts[3] == 1.0);

    // Matchpairs normal, low cost_unmatched, Inf forbidden pairs, huge scale (>1e15), NaN guard
    const keel::f64 cost[4] = {1.0, 100.0, 100.0, 2.0};
    auto mp = keel::geom::matchpairs({cost, 4}, 2, 2, 10.0, false);
    CHECK(mp.matches.size() == 2);
    CHECK(mp.unmatched_rows.empty() && mp.unmatched_cols.empty());
    CHECK(mp.matches[0].row == 0 && mp.matches[0].col == 0);
    CHECK(mp.matches[1].row == 1 && mp.matches[1].col == 1);

    auto mp2 = keel::geom::matchpairs({cost, 4}, 2, 2, 0.1, false);
    CHECK(mp2.matches.empty());
    CHECK(mp2.unmatched_rows.size() == 2 && mp2.unmatched_cols.size() == 2);

    // Inf forbidden off-diagonal entries
    const keel::f64 cost_inf[4] = {1.0, keel::kInf, keel::kInf, 2.0};
    auto mp_inf = keel::geom::matchpairs({cost_inf, 4}, 2, 2, 10.0, false);
    CHECK(mp_inf.matches.size() == 2);
    CHECK(mp_inf.matches[0].row == 0 && mp_inf.matches[0].col == 0);
    CHECK(mp_inf.matches[1].row == 1 && mp_inf.matches[1].col == 1);

    // Huge costs (> 1e15) must not collide with sentinel
    const keel::f64 cost_big[4] = {1e16, 1e18, 1e18, 2e16};
    auto mp_big = keel::geom::matchpairs({cost_big, 4}, 2, 2, 1e17, false);
    CHECK(mp_big.matches.size() == 2);
    CHECK(mp_big.matches[0].row == 0 && mp_big.matches[0].col == 0);
    CHECK(mp_big.matches[1].row == 1 && mp_big.matches[1].col == 1);

    // NaN cost_unmatched safely returns unmatched without crashing
    auto mp_nan = keel::geom::matchpairs({cost, 4}, 2, 2, keel::kNaN, false);
    CHECK(mp_nan.matches.empty() && mp_nan.unmatched_rows.size() == 2);
}

void dump_f64_array(const char* name, keel::ArrayRef<keel::f64> v) {
    keel::print("%s:", name);
    for (keel::usize i = 0; i < v.size(); ++i) {
        keel::print(" %.17g", v[i]);
    }
    keel::println("");
}

void dump_reference() {
    // 1. polyarea + inpolygon (interior, exterior, edges, vertices)
    const keel::f64 px[5] = {0.0, 3.0, 4.0, 1.5, -1.0};
    const keel::f64 py[5] = {0.0, 0.5, 3.0, 4.5, 2.0};
    keel::println("polyarea: %.17g", keel::geom::polyarea({px, 5}, {py, 5}));

    const keel::f64 qpx[6] = {1.5, 5.0, 0.0, 1.5, 3.5, -0.5};
    const keel::f64 qpy[6] = {2.0, 2.0, 0.0, 0.25, 1.75, 1.0};
    keel::u8 in_res[6] = {};
    keel::geom::inpolygon({qpx, 6}, {qpy, 6}, {px, 5}, {py, 5}, {in_res, 6});
    keel::print("inpolygon:");
    for (keel::usize i = 0; i < 6; ++i) keel::print(" %d", keel::as<keel::i32>(in_res[i]));
    keel::println("");

    // 2. convhull area + vertices
    const keel::f64 cx[10] = {0.0, 2.0, 4.0, 3.5, 1.0, -0.5, 1.5, 2.2, 0.8, 3.0};
    const keel::f64 cy[10] = {0.0, -0.8, 0.5, 3.2, 3.8, 1.5, 1.2, 1.8, 2.1, 1.0};
    auto ch = keel::geom::convhull({cx, 10}, {cy, 10});
    keel::Array<keel::f64> chx;
    keel::Array<keel::f64> chy;
    for (keel::usize i = 0; i + 1 < ch.size(); ++i) {
        chx.push_back(cx[ch[i]]);
        chy.push_back(cy[ch[i]]);
    }
    keel::println("convhull_area: %.17g", keel::geom::polyarea(chx.as_ref(), chy.as_ref()));
    keel::println("convhull_nverts: %d", keel::as<keel::i32>(ch.size() - 1));

    // 3. Scattered griddata (Linear, Nearest, Natural, Cubic, BiharmonicV4)
    keel::Array<keel::f64> sx;
    keel::Array<keel::f64> sy;
    keel::Array<keel::f64> sv;
    for (keel::usize i = 0; i < 5; ++i) {
        for (keel::usize j = 0; j < 5; ++j) {
            const keel::f64 x = keel::as<keel::f64>(i) + 0.07 * keel::sin(keel::as<keel::f64>(i * 3 + j));
            const keel::f64 y = keel::as<keel::f64>(j) + 0.07 * keel::cos(keel::as<keel::f64>(i + j * 5));
            sx.push_back(x);
            sy.push_back(y);
            sv.push_back(keel::sin(0.5 * x) * keel::cos(0.4 * y) + 0.2 * x - 0.1 * y);
        }
    }
    const keel::f64 qgx[4] = {0.75, 1.6, 2.35, 3.1};
    const keel::f64 qgy[4] = {0.85, 2.2, 1.45, 2.9};
    keel::f64 gout[4] = {};

    (void)keel::geom::griddata(sx.as_ref(), sy.as_ref(), sv.as_ref(), {qgx, 4}, {qgy, 4}, {gout, 4},
                               keel::geom::ScatteredMethod::Linear);
    dump_f64_array("griddata_linear", gout);
    (void)keel::geom::griddata(sx.as_ref(), sy.as_ref(), sv.as_ref(), {qgx, 4}, {qgy, 4}, {gout, 4},
                               keel::geom::ScatteredMethod::Nearest);
    dump_f64_array("griddata_nearest", gout);
    (void)keel::geom::griddata(sx.as_ref(), sy.as_ref(), sv.as_ref(), {qgx, 4}, {qgy, 4}, {gout, 4},
                               keel::geom::ScatteredMethod::Natural);
    dump_f64_array("griddata_natural", gout);
    (void)keel::geom::griddata(sx.as_ref(), sy.as_ref(), sv.as_ref(), {qgx, 4}, {qgy, 4}, {gout, 4},
                               keel::geom::ScatteredMethod::Cubic);
    dump_f64_array("griddata_cubic", gout);
    (void)keel::geom::griddata(sx.as_ref(), sy.as_ref(), sv.as_ref(), {qgx, 4}, {qgy, 4}, {gout, 4},
                               keel::geom::ScatteredMethod::BiharmonicV4);
    dump_f64_array("griddata_v4", gout);

    // 4. histcounts2
    const keel::f64 hx[8] = {0.1, 0.9, 1.0, 1.5, 2.0, 2.5, 0.5, 1.9};
    const keel::f64 hy[8] = {0.2, 0.8, 1.1, 1.9, 2.0, 0.5, 1.5, 0.1};
    const keel::f64 xe[3] = {0.0, 1.0, 2.0};
    const keel::f64 ye[3] = {0.0, 1.0, 2.0};
    keel::f64 hc[4] = {};
    keel::geom::histcounts2({hx, 8}, {hy, 8}, {xe, 3}, {ye, 3}, {hc, 4});
    dump_f64_array("histcounts2", hc);

    // 5. matchpairs (min, max, rectangular)
    // 3x3 col-major: [1 100 4; 100 2 7; 5 6 3] -> row-major C = [1 100 5; 100 2 6; 4 7 3]
    const keel::f64 cm[9] = {1.0, 100.0, 4.0, 100.0, 2.0, 7.0, 5.0, 6.0, 3.0};
    auto m_min = keel::geom::matchpairs({cm, 9}, 3, 3, 10.0, false);
    auto m_max = keel::geom::matchpairs({cm, 9}, 3, 3, 10.0, true);
    keel::f64 min_tot = 0.0;
    for (const auto& pr : m_min.matches) min_tot += cm[pr.col * 3 + pr.row];
    min_tot += 10.0 * keel::as<keel::f64>(m_min.unmatched_rows.size() + m_min.unmatched_cols.size());
    keel::f64 max_tot = 0.0;
    for (const auto& pr : m_max.matches) max_tot += cm[pr.col * 3 + pr.row];
    max_tot += 10.0 * keel::as<keel::f64>(m_max.unmatched_rows.size() + m_max.unmatched_cols.size());
    keel::println("matchpairs_min_obj: %.17g", min_tot);
    keel::println("matchpairs_max_obj: %.17g", max_tot);
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && keel::StringRef(argv[1]) == "--dump-ref") {
        dump_reference();
        return 0;
    }
    test_polygon_and_hull();
    test_delaunay_and_griddata();
    test_histcounts2_and_matchpairs();

    if (g_failed != 0) {
        keel::eprintln("test_geom: %d failure(s)", g_failed);
        return 1;
    }
    keel::println("test_geom: all tests passed");
    return 0;
}
