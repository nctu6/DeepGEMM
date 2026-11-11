#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include "../../runtime/jit.hpp"

namespace deep_gemm {

// `jit->compile` finds a loaded kernel by its source text: every call copies the compiler options, formats and hashes
// the flags, hashes the source and walks its includes, about 5 us of host time before the cache is even consulted.
// Launch paths that run for every layer of every step (the contiguous MQA logits of a prefill chunk) look the kernel up
// by the integers that fill its template instead, and build the source only on the first call of a configuration.
//
// `key` must hold every runtime value that appears in the source; a value the source ignores only splits an entry.
// The cache is per call site (one instantiation per `make_source` type). It is read and written with the GIL held, like
// the cache behind `jit->compile`.
struct KernelKeyHash {
    template <size_t N>
    size_t operator()(const std::array<int64_t, N>& key) const noexcept {
        uint64_t hash = 0x9e3779b97f4a7c15ull;
        for (const auto value: key)
            hash = (hash ^ static_cast<uint64_t>(value)) * 0x100000001b3ull + (hash >> 29);
        return static_cast<size_t>(hash);
    }
};

template <size_t N, typename MakeSource>
static auto get_cached_kernel(const std::string& name, const std::array<int64_t, N>& key, MakeSource&& make_source) {
    using KernelPtr = decltype(jit->compile(std::string(), std::string()));
    static std::unordered_map<std::array<int64_t, N>, KernelPtr, KernelKeyHash> cache;
    if (const auto iterator = cache.find(key); iterator != cache.end())
        return iterator->second;
    auto kernel = jit->compile(name, make_source());
    cache.emplace(key, kernel);
    return kernel;
}

} // namespace deep_gemm
