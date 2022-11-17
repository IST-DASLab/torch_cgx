/*
 * pytorch-cgx
 *
 * Copyright (C) 2022 Institute of Science and Technology Austria (ISTA).
 * All Rights Reserved.
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
#if QSGD_DETERMENISTIC
  return 0.5;
#else
  return ((float)xorshift128p(state_p)) / UINT64_MAX;
#endif
}


} // namespace gpu
} // namespace common
} // namespace cgx