// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <json.hpp>
#include <memory>
#include <sstream>
#include <stack>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/cuda.h"
#include "mlx/io.h"
#include "mlx/io/load.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

using json = nlohmann::json;

#define ST_F16 "F16"
#define ST_BF16 "BF16"
#define ST_F32 "F32"

#define ST_BOOL "BOOL"
#define ST_I8 "I8"
#define ST_I16 "I16"
#define ST_I32 "I32"
#define ST_I64 "I64"
#define ST_U8 "U8"
#define ST_U16 "U16"
#define ST_U32 "U32"
#define ST_U64 "U64"
#define ST_F8_E4M3 "F8_E4M3"
#define ST_F8_E8M0 "F8_E8M0"

// Note: Complex numbers aren't in the spec yet so this could change -
// https://github.com/huggingface/safetensors/issues/389
#define ST_C64 "C64"

namespace mlx::core {

std::string dtype_to_safetensor_str(Dtype t) {
  switch (t) {
    case float32:
      return ST_F32;
    case bfloat16:
      return ST_BF16;
    case float16:
      return ST_F16;
    case int64:
      return ST_I64;
    case int32:
      return ST_I32;
    case int16:
      return ST_I16;
    case int8:
      return ST_I8;
    case uint64:
      return ST_U64;
    case uint32:
      return ST_U32;
    case uint16:
      return ST_U16;
    case uint8:
      return ST_U8;
    case bool_:
      return ST_BOOL;
    case complex64:
      return ST_C64;
    default:
      throw std::runtime_error("[save_safetensors] received invalid dtype.");
  }
}

Dtype dtype_from_safetensor_str(std::string_view str) {
  if (str == ST_F32) {
    return float32;
  } else if (str == ST_F16) {
    return float16;
  } else if (str == ST_BF16) {
    return bfloat16;
  } else if (str == ST_I64) {
    return int64;
  } else if (str == ST_I32) {
    return int32;
  } else if (str == ST_I16) {
    return int16;
  } else if (str == ST_I8) {
    return int8;
  } else if (str == ST_U64) {
    return uint64;
  } else if (str == ST_U32) {
    return uint32;
  } else if (str == ST_U16) {
    return uint16;
  } else if (str == ST_U8) {
    return uint8;
  } else if (str == ST_BOOL) {
    return bool_;
  } else if (str == ST_C64) {
    return complex64;
  } else if (str == ST_F8_E4M3) {
    return uint8;
  } else if (str == ST_F8_E8M0) {
    return uint8;
  } else {
    std::ostringstream msg;
    msg << "[safetensor] unsupported dtype " << str;
    throw std::runtime_error(msg.str());
  }
}

namespace {

struct TensorEntry {
  std::string key;
  Dtype type;
  Shape shape;
  size_t begin; // absolute file offset of the first byte
  size_t nbytes;
};

struct SafetensorsHeader {
  std::vector<TensorEntry> tensors;
  std::unordered_map<std::string, std::string> metadata;
};

/** Read and validate the header; leaves the stream position unspecified */
SafetensorsHeader read_safetensors_header(io::Reader& in_stream) {
  ////////////////////////////////////////////////////////
  // Open and check file
  if (!in_stream.good() || !in_stream.is_open()) {
    std::ostringstream msg;
    msg << "[load_safetensors] Failed to open " << in_stream.label();
    throw std::runtime_error(msg.str());
  }

  uint64_t jsonHeaderLength = 0;
  // This is the same limit as in the original Rust Safetensors code.
  constexpr uint64_t kMaxJsonHeaderLength = 100000000;
  in_stream.read(reinterpret_cast<char*>(&jsonHeaderLength), 8);
  if (jsonHeaderLength <= 0 || jsonHeaderLength >= kMaxJsonHeaderLength) {
    std::ostringstream msg;
    msg << "[load_safetensors] Invalid json header length "
        << in_stream.label();
    throw std::runtime_error(msg.str());
  }

  // Determine file size to be able to validate that reads are within the
  // bounds of the file (at least at creation time)
  in_stream.seek(0, std::ios_base::end);
  size_t file_size = in_stream.tell();
  in_stream.seek(8, std::ios_base::beg);

  // Load the json metadata
  if (file_size < jsonHeaderLength + 8) {
    std::ostringstream msg;
    msg << "[load_safetensors] The JSON header is " << jsonHeaderLength
        << " bytes long but the file is only " << file_size << " bytes. "
        << "Perhaps an incomplete download or corrupt file?";
    throw std::runtime_error(msg.str());
  }
  auto rawJson = std::make_unique<char[]>(jsonHeaderLength);
  in_stream.read(rawJson.get(), jsonHeaderLength);
  auto metadata = json::parse(rawJson.get(), rawJson.get() + jsonHeaderLength);
  // Should always be an object on the top-level
  if (!metadata.is_object()) {
    std::ostringstream msg;
    msg << "[load_safetensors] Invalid json metadata " << in_stream.label();
    throw std::runtime_error(msg.str());
  }
  size_t offset = jsonHeaderLength + 8;

  SafetensorsHeader header;
  for (const auto& item : metadata.items()) {
    if (item.key() == "__metadata__") {
      for (const auto& meta_item : item.value().items()) {
        header.metadata.insert({meta_item.key(), meta_item.value()});
      }
      continue;
    }
    const std::string& dtype = item.value().at("dtype");
    const Shape& shape = item.value().at("shape");
    const std::vector<size_t>& data_offsets = item.value().at("data_offsets");
    Dtype type = dtype_from_safetensor_str(dtype);
    if (data_offsets.size() != 2) {
      std::ostringstream msg;
      msg << "[load_safetensors] Tensor '" << item.key()
          << "' data_offsets must have exactly 2 entries but has "
          << data_offsets.size();
      throw std::runtime_error(msg.str());
    }
    {
      size_t expected_nbytes = type.size();
      for (auto dim : shape) {
        expected_nbytes *= static_cast<size_t>(dim);
      }
      if (data_offsets[1] < data_offsets[0] ||
          data_offsets[1] - data_offsets[0] != expected_nbytes) {
        std::ostringstream msg;
        msg << "[load_safetensors] Tensor '" << item.key()
            << "' invalid data offsets (" << data_offsets[0] << ", "
            << data_offsets[1] << "). Expecting " << expected_nbytes
            << " bytes.";
        throw std::runtime_error(msg.str());
      }
    }
    if (offset + data_offsets[1] > file_size) {
      std::ostringstream msg;
      msg << "[load_safetensors] Tensor '" << item.key()
          << "' invalid data offsets (" << data_offsets[0] << ", "
          << data_offsets[1] << ") exceeding the size of the file. "
          << "Perhaps an incomplete download or corrupt file?";
      throw std::runtime_error(msg.str());
    }
    header.tensors.push_back(
        {item.key(),
         type,
         shape,
         offset + data_offsets[0],
         data_offsets[1] - data_offsets[0]});
  }
  return header;
}

array lazy_load(
    const TensorEntry& t,
    std::shared_ptr<io::Reader> in_stream,
    Stream stream) {
  return array(
      t.shape,
      t.type,
      std::make_shared<Load>(stream, in_stream, t.begin, false),
      std::vector<array>{});
}

#ifndef _WIN32
/** A read-only mapping of a whole file, unmapped with its last array */
struct FileMapping {
  void* ptr{MAP_FAILED};
  size_t size{0};
  ~FileMapping() {
    if (ptr != MAP_FAILED) {
      munmap(ptr, size);
    }
  }
};

std::shared_ptr<FileMapping> map_file(const std::string& file) {
  int fd = open(file.c_str(), O_RDONLY);
  if (fd < 0) {
    return nullptr;
  }
  auto mapping = std::make_shared<FileMapping>();
  off_t size = lseek(fd, 0, SEEK_END);
  if (size > 0) {
    mapping->size = static_cast<size_t>(size);
    mapping->ptr = mmap(nullptr, mapping->size, PROT_READ, MAP_SHARED, fd, 0);
  }
  close(fd);
  if (mapping->ptr == MAP_FAILED) {
    return nullptr;
  }
  return mapping;
}
#endif

} // namespace

/** Load array from reader in safetensor format */
SafetensorsLoad load_safetensors(
    std::shared_ptr<io::Reader> in_stream,
    StreamOrDevice s) {
  auto stream = cu::is_available() ? to_stream(s) : to_stream(s, Device::cpu);
  auto header = read_safetensors_header(*in_stream);

  std::unordered_map<std::string, array> res;
  for (const auto& t : header.tensors) {
    res.insert({t.key, lazy_load(t, in_stream, stream)});
  }
  return {res, std::move(header.metadata)};
}

SafetensorsLoad
load_safetensors(const std::string& file, bool zero_copy, StreamOrDevice s) {
  auto in_stream = std::make_shared<io::ParallelFileReader>(file);
  if (!zero_copy) {
    return load_safetensors(in_stream, s);
  }
  auto stream = cu::is_available() ? to_stream(s) : to_stream(s, Device::cpu);
  auto header = read_safetensors_header(*in_stream);

  // A tensor that starts on a page boundary is mapped instead of read; the
  // GPU then reads it straight from the page cache. Metal wraps only whole
  // pages, so the mapped range is rounded up to a page (the file must extend
  // that far; what lies past the tensor is never read). Other tensors, and
  // every tensor when the allocator cannot wrap the memory, load as usual.
  std::unordered_map<std::string, array> res;
#ifndef _WIN32
  const size_t page = static_cast<size_t>(getpagesize());
  std::shared_ptr<FileMapping> mapping;
  for (const auto& t : header.tensors) {
    if (t.nbytes > 0 && t.begin % page == 0) {
      mapping = map_file(file);
      break;
    }
  }
#endif
  for (const auto& t : header.tensors) {
#ifndef _WIN32
    size_t padded = (t.nbytes + page - 1) / page * page;
    if (mapping && t.nbytes > 0 && t.begin % page == 0 &&
        t.begin + padded <= mapping->size) {
      auto ptr = static_cast<char*>(mapping->ptr) + t.begin;
      auto buffer = allocator::make_buffer(ptr, padded);
      if (buffer.ptr() != nullptr) {
        array out(t.shape, t.type, nullptr, {});
        Strides strides(t.shape.size(), 1);
        for (int i = static_cast<int>(t.shape.size()) - 2; i >= 0; --i) {
          strides[i] = strides[i + 1] * t.shape[i + 1];
        }
        auto [no_bsx_size, row_contiguous, col_contiguous] =
            check_contiguity(t.shape, strides);
        // The deleter holds the mapping until the last mapped array is freed.
        out.set_data(
            allocator::Data(
                buffer,
                [mapping](allocator::Buffer b) { allocator::release(b); }),
            t.nbytes / t.type.size(),
            std::move(strides),
            array::Flags{true, row_contiguous, col_contiguous});
        out.set_status(array::Status::available);
        res.insert({t.key, std::move(out)});
        continue;
      }
    }
#endif
    res.insert({t.key, lazy_load(t, in_stream, stream)});
  }
  return {res, std::move(header.metadata)};
}

SafetensorsLoad load_safetensors(const std::string& file, StreamOrDevice s) {
  return load_safetensors(file, env::zero_copy_load(), s);
}

void save_safetensors(
    std::shared_ptr<io::Writer> out_stream,
    std::vector<std::pair<std::string, array>> a,
    std::unordered_map<std::string, std::string> metadata,
    bool page_align) {
  {
    std::vector<array> to_eval;
    to_eval.reserve(a.size());
    for (auto& p : a) {
      p.second = contiguous(p.second);
      to_eval.push_back(p.second);
    }
    eval(std::move(to_eval));
  }

  // Open only after the inputs are evaluated: opening a FileWriter
  // truncates the file and lazy inputs may still read from it.
  out_stream->open();

  ////////////////////////////////////////////////////////
  // Check file
  if (!out_stream->good() || !out_stream->is_open()) {
    std::ostringstream msg;
    msg << "[save_safetensors] Failed to open " << out_stream->label();
    throw std::runtime_error(msg.str());
  }

  // Whole-page tensors first: after a page-padded header each of them then
  // starts on a page boundary, and the rest follows with no gaps.
  size_t page = 1;
#ifndef _WIN32
  if (page_align) {
    page = static_cast<size_t>(getpagesize());
    auto rest = std::stable_partition(a.begin(), a.end(), [page](const auto& p) {
      return p.second.nbytes() > 0 && p.second.nbytes() % page == 0;
    });
    // The first of the other tensors also starts on a page: make it the
    // largest one (e.g. an embedding table); the small ones after it fill
    // its last page, so it can be mapped too.
    auto largest = std::max_element(rest, a.end(), [](const auto& x, const auto& y) {
      return x.second.nbytes() < y.second.nbytes();
    });
    if (largest != a.end()) {
      std::rotate(rest, largest, largest + 1);
    }
  }
#endif

  ////////////////////////////////////////////////////////
  // Check array map
  json parent;
  json _metadata;
  for (auto& [key, value] : metadata) {
    _metadata[key] = value;
  }
  parent["__metadata__"] = _metadata;

  size_t offset = 0;
  for (auto& [key, arr] : a) {
    json child;
    child["dtype"] = dtype_to_safetensor_str(arr.dtype());
    child["shape"] = arr.shape();
    child["data_offsets"] = std::vector<size_t>{offset, offset + arr.nbytes()};
    parent[key] = child;
    offset += arr.nbytes();
  }

  auto header = parent.dump();
  if (page > 1) {
    // The format allows trailing spaces in the header.
    header.append((page - (8 + header.length()) % page) % page, ' ');
  }
  uint64_t header_len = header.length();
  out_stream->write(reinterpret_cast<char*>(&header_len), 8);
  out_stream->write(header.c_str(), header_len);
  for (auto& [key, arr] : a) {
    // An empty tensor contributes a zero length span and has no data pointer
    // worth asking for.
    if (arr.nbytes() > 0) {
      out_stream->write(arr.data<char>(), arr.nbytes());
    }
  }
}

void save_safetensors(
    std::shared_ptr<io::Writer> out_stream,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  save_safetensors(
      std::move(out_stream),
      std::vector<std::pair<std::string, array>>(a.begin(), a.end()),
      std::move(metadata),
      false);
}

void save_safetensors(
    std::string file,
    std::vector<std::pair<std::string, array>> a,
    std::unordered_map<std::string, std::string> metadata,
    bool page_align) {
  // Add .safetensors to file name if it is not there
  if (file.length() < 12 ||
      file.substr(file.length() - 12, 12) != ".safetensors")
    file += ".safetensors";

  // Serialize array
  save_safetensors(
      std::make_shared<io::FileWriter>(std::move(file)),
      std::move(a),
      std::move(metadata),
      page_align);
}

void save_safetensors(
    std::string file,
    std::unordered_map<std::string, array> a,
    std::unordered_map<std::string, std::string> metadata /* = {} */) {
  save_safetensors(
      std::move(file),
      std::vector<std::pair<std::string, array>>(a.begin(), a.end()),
      std::move(metadata),
      false);
}

} // namespace mlx::core
