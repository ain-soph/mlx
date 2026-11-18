// Copyright © 2023-2024 Apple Inc.

#include "mlx/io.h"

namespace mlx::core {

SafetensorsLoad load_safetensors(std::shared_ptr<io::Reader>, StreamOrDevice) {
  throw std::runtime_error(
      "[load_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

SafetensorsLoad load_safetensors(const std::string&, StreamOrDevice) {
  throw std::runtime_error(
      "[load_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

void save_safetensors(
    std::shared_ptr<io::Writer>,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>) {
  throw std::runtime_error(
      "[save_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, std::string>) {
  throw std::runtime_error(
      "[save_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

SafetensorsLoad load_safetensors(const std::string&, bool, StreamOrDevice) {
  throw std::runtime_error(
      "[load_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

void save_safetensors(
    std::shared_ptr<io::Writer>,
    std::vector<std::pair<std::string, array>>,
    std::unordered_map<std::string, std::string>,
    bool) {
  throw std::runtime_error(
      "[save_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

void save_safetensors(
    std::string,
    std::vector<std::pair<std::string, array>>,
    std::unordered_map<std::string, std::string>,
    bool) {
  throw std::runtime_error(
      "[save_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

SafetensorsLoad load_safetensors(
    const std::string&,
    bool,
    const std::optional<std::vector<std::string>>&,
    StreamOrDevice) {
  throw std::runtime_error(
      "[load_safetensors] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}

bool zero_copy_load_default() {
  return false;
}

struct SafetensorsFile::Impl {};

SafetensorsFile::SafetensorsFile(const std::string&, bool, StreamOrDevice) {
  throw std::runtime_error(
      "[SafetensorsFile] Compile with MLX_BUILD_SAFETENSORS=ON "
      "to enable safetensors support.");
}
const std::vector<std::string>& SafetensorsFile::keys() const {
  throw std::runtime_error("[SafetensorsFile] not available");
}
const std::unordered_map<std::string, std::string>& SafetensorsFile::metadata()
    const {
  throw std::runtime_error("[SafetensorsFile] not available");
}
bool SafetensorsFile::contains(const std::string&) const {
  return false;
}
bool SafetensorsFile::mapped() const {
  return false;
}
array SafetensorsFile::get(const std::string&) const {
  throw std::runtime_error("[SafetensorsFile] not available");
}
std::optional<array> SafetensorsFile::get_rows(
    const std::vector<std::string>&) const {
  return std::nullopt;
}

} // namespace mlx::core
