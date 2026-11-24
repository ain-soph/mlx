// Copyright © 2023 Apple Inc.

#include <numeric>

#include "doctest/doctest.h"

#include "mlx/mlx.h"
#include "mlx/primitives.h"

using namespace mlx::core;

TEST_CASE("test matmul") {
  auto a = array(1);
  auto b = array({1.0});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = array({1.0});
  b = array({1.0});
  auto out = matmul(a, b);
  CHECK_EQ(out.shape(), Shape{});
  CHECK_EQ(out.size(), 1);
  CHECK_EQ(out.dtype(), float32);
  CHECK_EQ(out.item<float>(), 1.0f);

  a = ones({2, 4});
  b = ones({2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({3, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({4, 3, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 3});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4, 3});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({4, 2});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2, 2}, 4.0f)).item<bool>());

  a = ones({2, 4}, int32);
  b = ones({4, 2}, float32);
  out = matmul(a, b);
  CHECK(array_equal(out, full({2, 2}, 4.0f)).item<bool>());

  // Check single dimensions
  a = ones({4});
  b = ones({4, 2});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2}, 4.0f)).item<bool>());

  a = ones({2, 4});
  b = ones({4});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2}, 4.0f)).item<bool>());

  a = ones({4});
  b = ones({4});
  out = matmul(a, b);
  CHECK(array_equal(out, full({}, 4.0f)).item<bool>());

  // Test transposed arrays
  a = array({1.0f, 1.0f, 1.0f, 1.0f}, {1, 4});
  b = array({1.0f, 1.0f, 1.0f, 1.0f}, {4, 1});
  out = matmul(transpose(a), transpose(b));
  CHECK(array_equal(out, ones({4, 4})).item<bool>());

  a = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  b = array({1.0f, 2.0f, 1.0f, 2.0f}, {2, 2});
  out = matmul(transpose(a), b);
  CHECK(
      array_equal(out, array({4.0f, 8.0f, 6.0f, 12.0f}, {2, 2})).item<bool>());

  out = matmul(a, transpose(b));
  CHECK(
      array_equal(out, array({5.0f, 5.0f, 11.0f, 11.0f}, {2, 2})).item<bool>());

  out = matmul(transpose(a), transpose(b));
  CHECK(
      array_equal(out, array({7.0f, 7.0f, 10.0f, 10.0f}, {2, 2})).item<bool>());

  // Test broadcasting for both arrays
  a = ones({5, 4, 2});
  b = ones({2, 3});
  out = matmul(a, b);
  CHECK(array_equal(out, full({5, 4, 3}, 2.0f)).item<bool>());

  a = ones({5, 1, 4, 2});
  b = ones({1, 7, 2, 3});
  out = matmul(a, b);
  CHECK(array_equal(out, full({5, 7, 4, 3}, 2.0f)).item<bool>());

  // Test batched matmul with transpose
  a = ones({2, 2, 4});
  b = ones({2, 4, 2});
  out = matmul(transpose(a, {0, 2, 1}), transpose(b, {0, 2, 1}));
  CHECK(array_equal(out, full({2, 4, 4}, 2.0f)).item<bool>());
}

TEST_CASE("test mixed dtype matmul primitive on cpu") {
  // A graph exported from the GPU has no AsType before Matmul and AddMM.
  auto s = default_stream(Device::cpu);
  auto mixed_matmul = [&s](const array& a, const array& b) {
    Shape shape = a.shape();
    shape.back() = b.shape(-1);
    return array(shape, float32, std::make_shared<Matmul>(s), {a, b});
  };
  auto mixed_addmm = [&s](const array& c, const array& a, const array& b) {
    Shape shape = a.shape();
    shape.back() = b.shape(-1);
    return array(
        shape, float32, std::make_shared<AddMM>(s, 1.5f, 2.0f), {a, b, c});
  };
  auto ref_matmul = [&s](const array& a, const array& b) {
    return matmul(astype(a, float32, s), astype(b, float32, s), s);
  };
  auto close = [&s](const array& x, const array& y) {
    return allclose(x, y, 1e-5, 1e-5, false, s).item<bool>();
  };
  auto values = [&s](Shape shape, float scale) {
    int n = 1;
    for (auto d : shape) {
      n *= d;
    }
    auto x = multiply(astype(arange(n, s), float32, s), array(scale), s);
    return subtract(reshape(x, std::move(shape), s), array(1.0f), s);
  };

  for (auto [ta, tb] :
       {std::pair{bfloat16, float32},
        std::pair{float32, float16},
        std::pair{float16, bfloat16}}) {
    auto a = astype(values({4, 6}, 0.1f), ta, s);
    auto b = astype(values({6, 5}, 0.05f), tb, s);
    auto out = mixed_matmul(a, b);
    CHECK_EQ(out.dtype(), float32);
    CHECK(close(out, ref_matmul(a, b)));

    // Transposed and strided inputs
    auto at = transpose(astype(values({6, 4}, 0.1f), ta, s), s);
    auto bs = slice(
        astype(values({6, 10}, 0.05f), tb, s), {0, 0}, {6, 10}, {1, 2}, s);
    CHECK(close(mixed_matmul(at, bs), ref_matmul(at, bs)));

    // Batched
    auto ab = astype(values({2, 4, 6}, 0.1f), ta, s);
    auto bb = astype(values({2, 6, 5}, 0.05f), tb, s);
    CHECK(close(mixed_matmul(ab, bb), ref_matmul(ab, bb)));

    // AddMM
    auto c = values({4, 5}, 0.1f);
    auto expected =
        add(multiply(array(1.5f), ref_matmul(a, b), s),
            multiply(array(2.0f), c, s),
            s);
    CHECK(close(mixed_addmm(c, a, b), expected));

    // Empty outputs (M == 0 or N == 0) with K > 0
    auto a0 = astype(zeros({0, 6}, float32, s), ta, s);
    auto out0 = mixed_matmul(a0, bs);
    eval(out0);
    CHECK_EQ(out0.shape(), Shape{0, 5});
    auto b0 = astype(zeros({6, 0}, float32, s), tb, s);
    auto out1 = mixed_matmul(at, b0);
    eval(out1);
    CHECK_EQ(out1.shape(), Shape{4, 0});
  }
}
