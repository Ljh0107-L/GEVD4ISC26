#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace gevd4isc26::app {

[[nodiscard]] inline std::int64_t parseInteger(const std::string& text,
                                               const char* option) {
  std::size_t parsed_characters = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &parsed_characters);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("invalid integer for ") + option +
                                ": " + text);
  }
  if (parsed_characters != text.size()) {
    throw std::invalid_argument(std::string("invalid integer for ") + option +
                                ": " + text);
  }
  return static_cast<std::int64_t>(value);
}

[[nodiscard]] inline std::int64_t integerOption(
    int argc,
    char** argv,
    const std::string& name,
    std::int64_t fallback) {
  for (int index = 1; index < argc; ++index) {
    if (argv[index] != name) {
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value after " + name);
    }
    return parseInteger(argv[index + 1], name.c_str());
  }
  return fallback;
}

[[nodiscard]] inline bool hasFlag(int argc,
                                  char** argv,
                                  const std::string& name) {
  for (int index = 1; index < argc; ++index) {
    if (argv[index] == name) {
      return true;
    }
  }
  return false;
}

}  // namespace gevd4isc26::app
