namespace cgx {
namespace common {
namespace gpu {

inline __device__ uint64_t splitmix64(uint64_t* seed) {
  uint64_t result = *seed;

  *seed = result + 0x9E3779B97f4A7C15;
  result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
  result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
  return result ^ (result >> 31);
}

inline __device__ xorshift128p_state xorshift128_init(uint64_t seed) {
  xorshift128p_state result;
  uint64_t tmp = splitmix64(&seed);
  result.a = tmp;
  tmp = splitmix64(&seed);
  result.b = tmp;
  return result;
}


inline __device__ float xorshift128p(xorshift128p_state* state) {
  uint64_t t = state->a;
  uint64_t s = state->b;
  state->a = s;
  t ^= t << 23;       // a
  t ^= t >> 17;       // b
  t ^= s ^ (s >> 26); // c
  state->b = t;
  return (t + s) * 1.0;
}

__device__ float GetRand(RandState* state_p) {
  return ((float)xorshift128p(state_p)) / UINT64_MAX;
}


} // namespace gpu
} // namespace common
} // namespace cgx