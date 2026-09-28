// SPDX-License-Identifier: Apache-2.0
// NIRNAY - build identification. The macros are supplied by CMake.

#include "nirnay/version.hpp"

#include <string>

#include <fmt/format.h>

#ifndef NIRNAY_VERSION
#define NIRNAY_VERSION "0.0.0-unconfigured"
#endif
#ifndef NIRNAY_GIT_COMMIT
#define NIRNAY_GIT_COMMIT "unknown"
#endif
#ifndef NIRNAY_BUILD_TYPE
#define NIRNAY_BUILD_TYPE "unknown"
#endif
#ifndef NIRNAY_COMPILER
#define NIRNAY_COMPILER "unknown"
#endif

namespace nirnay {

const char* version_string() noexcept {
  return NIRNAY_VERSION;
}
const char* git_commit() noexcept {
  return NIRNAY_GIT_COMMIT;
}
const char* build_type() noexcept {
  return NIRNAY_BUILD_TYPE;
}
const char* compiler_string() noexcept {
  return NIRNAY_COMPILER;
}

bool cuda_enabled() noexcept {
#ifdef NIRNAY_ENABLE_CUDA
  return true;
#else
  return false;
#endif
}

const char* banner() noexcept {
  static const std::string text =
      fmt::format("NIRNAY {} ({}, {}, {}, CUDA {})", version_string(), git_commit(),
                  build_type(), compiler_string(), cuda_enabled() ? "on" : "off");
  return text.c_str();
}

}  // namespace nirnay
