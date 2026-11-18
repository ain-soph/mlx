// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <unordered_map>
#include <variant>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/io/load.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core {
using GGUFMetaData =
    std::variant<std::monostate, array, std::string, std::vector<std::string>>;
using GGUFLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, GGUFMetaData>>;
using SafetensorsLoad = std::pair<
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>>;

/** Save array to out stream in .npy format */
MLX_API void save(std::shared_ptr<io::Writer> out_stream, array a);

/** Save array to file in .npy format */
MLX_API void save(std::string file, array a);

/** Load array from reader in .npy format */
MLX_API array
load(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s = {});

/** Load array from file in .npy format */
MLX_API array load(std::string file, StreamOrDevice s = {});

/** Load array map from .safetensors file format */
MLX_API SafetensorsLoad
load_safetensors(std::shared_ptr<io::Reader> in_stream, StreamOrDevice s = {});
MLX_API SafetensorsLoad
load_safetensors(const std::string& file, StreamOrDevice s = {});
/**
 * Load from a .safetensors file. With zero_copy, tensors that start on a page
 * boundary and span whole pages are mapped read-only from the file instead of
 * read into new buffers (see save_safetensors with page_align). The overload
 * without the flag uses the MLX_ZERO_COPY_LOAD environment variable.
 */
MLX_API SafetensorsLoad load_safetensors(
    const std::string& file,
    bool zero_copy,
    StreamOrDevice s = {});

MLX_API void save_safetensors(
    std::shared_ptr<io::Writer> in_stream,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> metadata = {});
MLX_API void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string> metadata = {});
/**
 * Save arrays in the given order. With page_align, the header is padded to a
 * page and the tensors whose size is a whole number of pages come first, so
 * each of them starts on a page boundary and can be loaded zero-copy. The
 * file stays an ordinary, gap-free safetensors file.
 */
MLX_API void save_safetensors(
    std::shared_ptr<io::Writer> out_stream,
    std::vector<std::pair<std::string, array>> arrays,
    std::unordered_map<std::string, std::string> metadata,
    bool page_align);
MLX_API void save_safetensors(
    std::string file,
    std::vector<std::pair<std::string, array>> arrays,
    std::unordered_map<std::string, std::string> metadata,
    bool page_align);

/** Load array map and metadata from .gguf file format */

MLX_API GGUFLoad load_gguf(const std::string& file, StreamOrDevice s = {});

MLX_API void save_gguf(
    std::string file,
    std::unordered_map<std::string, array> array_map,
    std::unordered_map<std::string, GGUFMetaData> meta_data = {});

} // namespace mlx::core
