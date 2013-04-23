#ifndef ADVCL_CLL_H_INCLUDED
#define ADVCL_CLL_H_INCLUDED

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

// Issue with using cl_float4 from cl_platform.h.
// http://www.khronos.org/message_boards/viewtopic.php?f=28&t=1848
// typedef cl_float cl_float4 __attribute__ ((__vector_size__ (16), __may_alias__));
typedef struct Vec4 {
  float x,y,z,w;
  Vec4(){};
  // Convenience functions.
  Vec4(float xx, float yy, float zz, float ww):
    x(xx),
    y(yy),
    z(zz),
    w(ww)
  {}
  void set(float xx, float yy, float zz, float ww=1.) {
    x = xx;
    y = yy;
    z = zz;
    w = ww;
  }
} Vec4; // __attribute__((aligned(16)));

class MD {
public:
  // These are arrays we will use in this tutorial.
  std::vector<cl::Memory> cl_vbos;  // 0: position vbo, 1: color vbo.
  cl::Buffer cl_velocities;  // Particle velocities.
  cl::Buffer cl_pos_gen;  // want to have the start points for reseting particles
  cl::Buffer cl_vel_gen;  // want to have the start velocities for reseting particles

  GLuint p_vbo;          // Position vbo.
  GLuint c_vbo;          // Colors vbo.
  int num;            // The number of particles.
  size_t array_size;  // The size of our arrays num * sizeof(Vec4).

  // Default constructor initializes OpenCL context and automatically chooses
  // platform and device.
  MD();
  // Default destructor releases OpenCL objects and frees device memory.
  ~MD();

  std::string loadFile(const char *filename);
  // Load an OpenCL program from a string.
  void loadProgram(std::string kernel_source);
  void loadData(std::vector<Vec4> pos, std::vector<Vec4> vel,
                std::vector<Vec4> col);
  // These are implemented in part1.cpp (in the future we will make these more
  // general).
  void popCorn();
  // Execute the kernel.
  void runKernel();

private:

  unsigned int deviceUsed;
  std::vector<cl::Device> devices;

  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;
  cl::Kernel kernel;

  // Debugging variables.
  cl_int err;
  /// cl_event event;
  cl::Event event;
};

#endif
