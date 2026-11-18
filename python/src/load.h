// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include "mlx/io.h"

namespace mx = mlx::core;
namespace nb = nanobind;

using LoadOutputTypes = std::variant<
    mx::array,
    std::unordered_map<std::string, mx::array>,
    mx::SafetensorsLoad,
    mx::GGUFLoad>;

mx::SafetensorsLoad mlx_load_safetensor_helper(
    nb::object file,
    mx::StreamOrDevice s,
    std::optional<bool> zero_copy = std::nullopt,
    std::optional<std::vector<std::string>> keys = std::nullopt);
void mlx_save_safetensor_helper(
    nb::object file,
    nb::dict d,
    std::optional<nb::dict> m,
    bool page_align = false);

mx::GGUFLoad mlx_load_gguf_helper(nb::object file, mx::StreamOrDevice s);

void mlx_save_gguf_helper(
    nb::object file,
    nb::dict d,
    std::optional<nb::dict> m);

LoadOutputTypes mlx_load_helper(
    nb::object file,
    std::optional<std::string> format,
    bool return_metadata,
    std::optional<bool> zero_copy,
    std::optional<std::vector<std::string>> keys,
    mx::StreamOrDevice s);
void mlx_save_helper(nb::object file, mx::array a);
void mlx_savez_helper(
    nb::object file,
    nb::args args,
    const nb::kwargs& kwargs,
    bool compressed = false);
