#include <cstdint>
#include <cstring>

namespace {

float bits_to_float(uint32_t bits) {
  float value;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t float_to_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint16_t float_to_bf16(float value) {
  const uint32_t bits = float_to_bits(value);
  const uint32_t exponent = bits & 0x7f800000u;
  const uint32_t mantissa = bits & 0x007fffffu;
  uint32_t upper = bits >> 16;
  if (exponent == 0x7f800000u && mantissa != 0) {
    return static_cast<uint16_t>(upper | 0x0040u);
  }
  upper = (bits + 0x7fffu + (upper & 1u)) >> 16;
  return static_cast<uint16_t>(upper);
}

float bf16_to_float(uint16_t bits) {
  return bits_to_float(static_cast<uint32_t>(bits) << 16);
}

}  // namespace

extern "C" uint32_t spinquant_fp32_mul_bits(uint32_t a, uint32_t b) {
  volatile float result = bits_to_float(a) * bits_to_float(b);
  return float_to_bits(result);
}

extern "C" uint32_t spinquant_fp32_add_bits(uint32_t a, uint32_t b) {
  volatile float result = bits_to_float(a) + bits_to_float(b);
  return float_to_bits(result);
}

extern "C" uint32_t spinquant_i32_to_fp32_bits(int32_t a) {
  volatile float result = static_cast<float>(a);
  return float_to_bits(result);
}

extern "C" uint32_t spinquant_bf16_mul_bits(uint32_t a, uint32_t b) {
  volatile float result = bf16_to_float(static_cast<uint16_t>(a)) *
                          bf16_to_float(static_cast<uint16_t>(b));
  return float_to_bf16(result);
}

extern "C" uint32_t spinquant_i32_to_bf16_bits(int32_t a) {
  volatile float result = static_cast<float>(a);
  return float_to_bf16(result);
}
