#include "keel/geom.h"
#include "keel/math.h"

#include <cstdio>

namespace {

keel::i32 g_failed = 0;

void check(bool cond, const char* expr, keel::i32 line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [line %d]: %s\n", keel::as<int>(line), expr);
        ++g_failed;
    }
}

void check_near(keel::f64 got, keel::f64 expected, keel::f64 tol, const char* expr, keel::i32 line) {
    if (keel::is_nan(got) || keel::abs(got - expected) > tol) {
        std::fprintf(stderr, "FAIL [line %d]: %s (got %.15g, expected %.15g)\n",
                     keel::as<int>(line), expr, got, expected);
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

    const keel::f64 px[6] = {0.0, 2.0, 2.0, 0.0, 1.0, 0.5};
    const keel::f64 py[6] = {0.0, 0.0, 2.0, 2.0, 1.0, 1.2};
    auto hull = keel::geom::convhull({px, 6}, {py, 6});
    CHECK(hull.size() == 5);
    CHECK(hull[0] == hull[4]);

    auto bnd = keel::geom::boundary({px, 6}, {py, 6}, 0.0);
    CHECK(bnd.size() == 5);
}

void test_delaunay_and_griddata() {
    // Small cloud (<= 30): square + center -> 4 triangles.
    const keel::f64 x[5] = {0.0, 1.0, 1.0, 0.0, 0.5};
    const keel::f64 y[5] = {0.0, 0.0, 1.0, 1.0, 0.5};
    auto tris = keel::geom::delaunay({x, 5}, {y, 5});
    CHECK(tris.size() == 4);

    // Larger cloud (> 30) exercises Bowyer-Watson path: 7x7 grid = 49 points -> 72 triangles.
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
    CHECK(big_tris.size() >= 60);

    // Linear griddata reproduces affine function exactly inside hull, NaN outside.
    const keel::f64 qx[2] = {2.5, 100.0};
    const keel::f64 qy[2] = {3.5, 100.0};
    keel::f64 qout[2] = {0.0, 0.0};
    CHECK(keel::geom::griddata(gx.as_ref(), gy.as_ref(), gv.as_ref(), {qx, 2}, {qy, 2}, {qout, 2},
                         keel::geom::ScatteredMethod::Linear));
    CHECK_NEAR(qout[0], 2.0 * 2.5 + 3.0 * 3.5 + 1.0, 1e-10);
    CHECK(keel::is_nan(qout[1]));

    // Nearest & BiharmonicV4.
    const keel::f64 v5[5] = {0.0, 1.0, 2.0, 1.0, 1.0};
    const keel::f64 q1x[1] = {0.5};
    const keel::f64 q1y[1] = {0.5};
    keel::f64 q1out[1] = {0.0};
    CHECK(keel::geom::griddata({x, 5}, {y, 5}, {v5, 5}, {q1x, 1}, {q1y, 1}, {q1out, 1},
                         keel::geom::ScatteredMethod::Nearest));
    CHECK_NEAR(q1out[0], 1.0, 1e-14);
    CHECK(keel::geom::griddata({x, 5}, {y, 5}, {v5, 5}, {q1x, 1}, {q1y, 1}, {q1out, 1},
                         keel::geom::ScatteredMethod::BiharmonicV4));
    CHECK_NEAR(q1out[0], 1.0, 1e-10);

    // ND nearest.
    const keel::f64 nd_x[6] = {0.0, 10.0, 0.0, 10.0, 0.0, 10.0}; // 2x3 col-major
    const keel::f64 nd_v[2] = {42.0, 99.0};
    const keel::f64 nd_q[3] = {9.0, 9.0, 9.0};
    keel::f64 nd_out[1] = {0.0};
    CHECK(keel::geom::griddatan_nearest({nd_x, 6}, 2, 3, {nd_v, 2}, {nd_q, 3}, 1, {nd_out, 1}));
    CHECK_NEAR(nd_out[0], 99.0, 1e-14);
}

void test_histcounts2_and_matchpairs() {
    const keel::f64 hx[4] = {0.5, 1.5, 2.0, 10.0};
    const keel::f64 hy[4] = {0.5, 0.5, 2.0, 0.5};
    const keel::f64 xe[3] = {0.0, 1.0, 2.0};
    const keel::f64 ye[3] = {0.0, 1.0, 2.0};
    keel::f64 counts[4] = {0.0, 0.0, 0.0, 0.0};
    keel::geom::histcounts2({hx, 4}, {hy, 4}, {xe, 3}, {ye, 3}, {counts, 4});
    CHECK(counts[0] == 1.0); // (0,0)
    CHECK(counts[1] == 1.0); // (1,0)
    CHECK(counts[2] == 0.0); // (0,1)
    CHECK(counts[3] == 1.0); // (1,1), right edge inclusive

    // Matchpairs: 2x2 cost matrix [1, 100; 100, 2] col-major -> matches (0,0) and (1,1).
    const keel::f64 cost[4] = {1.0, 100.0, 100.0, 2.0};
    auto mp = keel::geom::matchpairs({cost, 4}, 2, 2, 10.0, false);
    CHECK(mp.matches.size() == 2);
    CHECK(mp.unmatched_rows.empty());
    CHECK(mp.unmatched_cols.empty());
    CHECK(mp.matches[0].row == 0 && mp.matches[0].col == 0);
    CHECK(mp.matches[1].row == 1 && mp.matches[1].col == 1);

    // Low cost_unmatched leaves expensive entries unmatched.
    auto mp2 = keel::geom::matchpairs({cost, 4}, 2, 2, 0.1, false);
    CHECK(mp2.matches.empty());
    CHECK(mp2.unmatched_rows.size() == 2);
    CHECK(mp2.unmatched_cols.size() == 2);
}

} // namespace

int main() {
    test_polygon_and_hull();
    test_delaunay_and_griddata();
    test_histcounts2_and_matchpairs();

    if (g_failed != 0) {
        std::fprintf(stderr, "test_geom: %d failure(s)\n", keel::as<int>(g_failed));
        return 1;
    }
    std::printf("test_geom: all tests passed\n");
    return 0;
}
