#define STRINGIFY(A) #A

__kernel void lj_pot(float3 pos1, float3 pos2, __local float3* res) {
  float sigma = 4.10;
  float epsilon = 1.77;

  float3 dist = distance(pos1, pos2);
  float sigr = sigma / length(dist);

  float lj = 4 * epsilon * (pow(sigr, 12) - pow(sigr, 6));
  lj = 0.f;

  res->x = lj * dist.x;
  res->y = lj * dist.y;
  res->z = lj * dist.z;
}

__kernel void force(__global float3* pos, __global float4* color,
                    __global float3* force, __global float3* pos_gen,
                    int num) {
  // Get our index in the array.
  size_t idx = get_global_id(0);
  // Copy position and velocity for this iteration to a local variable.
  // Note: if we were doing many more calculations we would want to have opencl
  // copy to a local memory array to speed up memory access (this will be the
  // subject of a later tutorial)
  float3 p = pos[idx];
  __local float3 lj;
  float3 f;
  f.x = 0.f; f.y = 0.f; f.z = 0.f;

  for (int i = idx + 1; i < num; i++) {
    lj_pot(p, pos[i], &lj);
    f += lj;
  }

  force[idx] = f;
}

//std::string kernel_source = STRINGIFY(
__kernel void update(__global float3* pos, __global float4* color,
                     __global float3* force, __global float3* vel,
                     float dt) {
  // Get our index in the array.
  size_t i = get_global_id(0);
  // Copy position and velocity for this iteration to a local variable.
  // Note: if we were doing many more calculations we would want to have opencl
  // copy to a local memory array to speed up memory access (this will be the
  // subject of a later tutorial)
  float3 p = pos[i];
  float3 f = force[i];
  float3 a = f / 1.f;
  float3 v = vel[i];
  v += a * dt;
  p += v * dt;

  vel[i] = v;
  pos[i] = p;
}
//);
