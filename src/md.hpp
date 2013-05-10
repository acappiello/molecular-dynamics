#ifndef ADVCL_CLL_H_INCLUDED
#define ADVCL_CLL_H_INCLUDED

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"


class MD {
public:
  // These are arrays used by the GPU.
  std::vector<cl::Memory> cl_vbos;  // 0: position vbo, 1: color vbo.
  cl::Buffer cl_forces;
  cl::Buffer cl_vel;

  GLuint pos_vbo;     // Position vbo.
  GLuint col_vbo;     // Colors vbo.
  int num;            // The number of particles.
  size_t array_size;  // The size of our arrays num * sizeof(cl_float4).

  // Default constructor initializes OpenCL context and automatically chooses
  // platform and device.
  MD();
  // Default destructor. Currently does nothing because the program ends.
  ~MD();

  std::string loadFile(const char *filename);
  // Load an OpenCL program from a string.
  void loadProgram(std::string kernel_source, int group_size_val);
  void loadData(std::vector<cl_float4> pos, std::vector<cl_float4> force,
                std::vector<cl_float4> vel, std::vector<cl_float4> col);
  void clInit(float bound, float dt, std::string force_kernel_name);
  // Execute the kernel.
  void runKernel();

private:

  unsigned int deviceUsed;
  std::vector<cl::Device> devices;

  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;
  cl::Kernel kernel;
  cl::Kernel forceKernel;
  cl::Kernel updateKernel;

  int group_size;

  // Debugging variables.
  cl_int err;
  /// cl_event event;
  cl::Event event;
};

#endif
