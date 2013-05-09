#define STRINGIFY(A) #A

__kernel void lj_pot(float4 pos1, float4 pos2, __local float4* res) {
  float sigma = 4.10;
  float epsilon = 1.77;

  float dist = distance(pos1, pos2);
  if (dist < 0.5) dist = 0.5;
  float sigr = sigma / dist;

  float lj = 4 * epsilon * (pow(sigr, 12) - pow(sigr, 6));

  res->x = lj * (pos1.x - pos2.x);
  res->y = lj * (pos1.y - pos2.y);
  res->z = lj * (pos1.z - pos2.z);
  res->w = 0.f;
}

__kernel void force(__global float4* pos, __global float4* color,
                    __global float4* force, __global float4* pos_gen,
                    int num) {
  // Get our index in the array.
  size_t idx = get_global_id(0);
  // Copy position and velocity for this iteration to a local variable.
  // Note: if we were doing many more calculations we would want to have opencl
  // copy to a local memory array to speed up memory access (this will be the
  // subject of a later tutorial)
  float4 p = pos[idx];
  __local float4 lj;
  float4 f;
  f.x = 0.f; f.y = 0.f; f.z = 0.f; f.w = 0.f;

  for (int i = 0; i < num; i++) {
    if (i != idx) {
      lj_pot(p, pos[i], &lj);
      f += lj;
    }
  }

  force[idx] = f;
}

//std::string kernel_source = STRINGIFY(
__kernel void update(__global float4* pos, __global float4* color,
                     __global float4* force, __global float4* vel,
                     float bound, float dt) {
  // Get our index in the array.
  size_t i = get_global_id(0);
  // Copy position and velocity for this iteration to a local variable.
  // Note: if we were doing many more calculations we would want to have opencl
  // copy to a local memory array to speed up memory access (this will be the
  // subject of a later tutorial)
  float4 p = pos[i];
  float4 f = force[i];
  float4 a = f / 1.f;
  float4 v = vel[i];
  v += a * dt;
  p += v * dt;

  float elasticity = 0.5;
  if (p.x >= bound || p.x <= -bound)
    v.x = -elasticity * v.x;
  if (p.y >= bound || p.y <= -bound)
    v.y = -elasticity * v.y;
  if (p.z >= bound || p.z <= -bound)
    v.z = -elasticity * v.z;

  p = clamp(p, -bound, bound);
  pos[i] = p;
  vel[i] = v;

  color[i] = clamp((p + bound) / (2 * bound), 0.2f, 1.f);
}
//);
