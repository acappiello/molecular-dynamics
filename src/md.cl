#define ZERO4 ((float4)(0.f, 0.f, 0.f, 0.f))

float4 lj_pot(float4 pos1, float4 pos2, float dist) {
  float4 res;
  float sigma = 4.10;
  float epsilon = 1.77;

  if (dist <= 1e-5)
    return ZERO4;   // Don't count yourself.
  if (dist <= 0.1)
    dist = 0.1;     // Particles can never be on top of each other!
  float sigr = sigma / dist;

  float lj = 4 * epsilon * (pow(sigr, 12) - pow(sigr, 6));

  res = lj * (pos1 - pos2);
  return res;
}

__kernel void force_naive(__global float4* pos, __global float4* color,
                          __global float4* force, int num) {
  // Get our index in the array.
  size_t idx = get_global_id(0);
  // Copy position for this iteration to a local variable.
  float4 p = pos[idx];
  float4 f = ZERO4;
  float cutoff = 10.f;

  for (int i = 0; i < num; i++) {
    if (i != idx)
      f += lj_pot(p, pos[i], distance(p, pos[i]));
  }

  force[idx] = f;
}

__kernel void force_tile(__global float4* pos, __global float4* color,
                         __global float4* force, int num) {
  // Get our index in the array.
  size_t ix = get_group_id(0);
  size_t lx = get_local_id(0);
  size_t l_dim = get_local_size(0);
  size_t idx = ix * l_dim + lx;
  // Copy position for this iteration to a local variable.
  float4 p = pos[idx];
  float4 f = (float4)(0.f, 0.f, 0.f, 0.f);
  float cutoff = 10.f;

  __local float4 workspace[SIZE];

  int tile = 0;
  for (int i = 0; i < num; i+=SIZE) {
    int id = tile * l_dim + lx;
    workspace[lx] = pos[id];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = 0; j < l_dim; j++) {
      f += lj_pot(p, workspace[j], distance(p, workspace[j]));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    tile++;
  }

  force[idx] = f;
}

__kernel void update(__global float4* pos, __global float4* color,
                     __global float4* force, __global float4* vel,
                     float bound, float dt) {
  // Get our index in the array.
  size_t i = get_global_id(0);
  // Copy position, velocity, and force for this iteration to a local variable.
  float4 p = pos[i];
  float4 f = force[i];
  float4 a = f / 1.f;  // Let's just call the mass 1.
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
  color[i].w = 1.f;  // Leave alpha alone.
}
