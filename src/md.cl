#define STRINGIFY(A) #A

//std::string kernel_source = STRINGIFY(
__kernel void update(__global float4* pos, __global float4* color,
                    __global float4* vel, __global float4* pos_gen,
                    __global float4* vel_gen, float dt) {
  // Get our index in the array.
  unsigned int i = get_global_id(0);
  // Copy position and velocity for this iteration to a local variable.
  // Note: if we were doing many more calculations we would want to have opencl
  // copy to a local memory array to speed up memory access (this will be the
  // subject of a later tutorial)
  float4 p = pos[i];
  float4 v = vel[i];

  // We've stored the life in the fourth component of our velocity array.
  float life = vel[i].w;
  // Decrease the life by the time step (this value could be adjusted to
  // lengthen or shorten particle life
  life -= dt;
  // If the life is 0 or less we reset the particle's values back to the
  // original values and set life to 1
  if(life <= 0) {
    p = pos_gen[i];
    v = vel_gen[i];
    life = 1.0;
  }

  // We use a first order euler method to integrate the velocity and position
  // (i'll expand on this in another tutorial).
  // Update the velocity to be affected by "gravity" in the z direction.
  v.z -= 9.8*dt;
  // Update the position with the new velocity.
  p.z += v.z*dt;
  // Store the updated life in the velocity array.
  v.w = life;

  // Update the arrays with our newly computed values.
  pos[i] = p;
  vel[i] = v;

  // You can manipulate the color based on properties of the system.
  // Here we adjust the alpha.
  color[i].w = life;
}
//);
