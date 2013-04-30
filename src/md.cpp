#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <iterator>

#include "md.hpp"
#include "util.hpp"
#include "types.hpp"

// Needed for context sharing functions.
#include <GL/glx.h>


MD::MD() {
  printf("Initialize OpenCL object and context\n");
  // Setup devices and context.
  std::vector<cl::Platform> platforms;
  err = cl::Platform::get(&platforms);
  printf("cl::Platform::get(): %s\n", oclErrorString(err));
  printf("platforms.size(): %lu\n", platforms.size());

  deviceUsed = 0;
  err = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
  printf("getDevices: %s\n", oclErrorString(err));
  printf("devices.size(): %lu\n", devices.size());
  int t = devices.front().getInfo<CL_DEVICE_TYPE>();
  printf("type: device: %d CL_DEVICE_TYPE_GPU: %d \n", t, CL_DEVICE_TYPE_GPU);

  // Define OS-specific context properties and create the OpenCL context.
  // We setup OpenGL context sharing slightly differently on each OS.
  // This code comes mostly from NVIDIA's SDK examples.
  // We could also check to see if the device supports GL sharing
  // but that is just searching through the properties.
  // An example is avaible in the NVIDIA code.
  cl_context_properties props[] =
    {
      CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
      CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(),
      0
    };
  // cl_context cxGPUContext = clCreateContext(props, 1,
  //            &cdDevices[uiDeviceUsed], NULL, NULL, &err);
  try {
    context = cl::Context(CL_DEVICE_TYPE_GPU, props);
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
  }

  // Create the command queue we will use to execute OpenCL commands.
  try {
    queue = cl::CommandQueue(context, devices[deviceUsed], 0, &err);
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%d)\n", er.what(), er.err());
  }
}

MD::~MD()
{}

std::string MD::loadFile(const char *filename) {
  std::ifstream file(filename);
  std::string prog(std::istreambuf_iterator<char>(file),
                   (std::istreambuf_iterator<char>()));
  return prog;
}

void MD::loadProgram(std::string kernel_source) {
  // Program Setup.
  int pl;
  //size_t program_length;
  printf("load the program\n");
  bool failed = false;

  pl = kernel_source.size();
  printf("kernel size: %d\n", pl);
  //printf("kernel: \n %s\n", kernel_source.c_str());
  try {
    cl::Program::Sources source(1,
                                std::make_pair(kernel_source.c_str(), pl));
    program = cl::Program(context, source);
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
  }

  printf("build program\n");
  try {
    //err = program.build(devices, "-cl-nv-verbose -cl-nv-maxrregcount=100");
    err = program.build(devices);
  }
  catch (cl::Error er) {
    printf("program.build: %s\n", oclErrorString(er.err()));
    //if(err != CL_SUCCESS){
    failed = true;
  }
  printf("done building program\n");
  std::cout << "Build Status: "
            << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0])
            << std::endl;
  std::cout << "Build Options:\t"
            << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0])
            << std::endl;
  std::cout << "Build Log:\t "
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
            << std::endl;

  if (failed)
    exit(EXIT_FAILURE);
}

void MD::loadData(std::vector<cl_float4> pos, std::vector<cl_float4> force,
                  std::vector<cl_float4> vel, std::vector<cl_float4> col) {
  // Store the number of particles and the size in bytes of our arrays.
  num = pos.size();
  array_size = num * sizeof(cl_float4);
  // Create VBOs (defined in util.cpp).
  pos_vbo = createVBO(&pos[0], array_size, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);
  col_vbo = createVBO(&col[0], array_size, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW);

  // Make sure OpenGL is finished before we proceed.
  glFinish();
  printf("gl interop!\n");
  // Create OpenCL buffer from GL VBO.
  cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_WRITE, pos_vbo, &err));
  printf("cl::BufferGL: %s\n", oclErrorString(err));
  //printf("v_vbo: %s\n", oclErrorString(err));
  cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_WRITE, col_vbo, &err));
  printf("cl::BufferGL: %s\n", oclErrorString(err));
  // We don't need to push any data here because it's already in the VBO.

  // Create the OpenCL only arrays.
  cl_pos_gen = cl::Buffer(context, CL_MEM_WRITE_ONLY, array_size, NULL, &err);
  cl_forces = cl::Buffer(context, CL_MEM_READ_WRITE, array_size, NULL, &err);
  cl_vel = cl::Buffer(context, CL_MEM_READ_WRITE, array_size, NULL, &err);

  printf("Pushing data to the GPU\n");
  // Push our CPU arrays to the GPU.
  // Data is tightly packed in std::vector starting with the adress of the first
  // element.
  err = queue.enqueueWriteBuffer(cl_forces, CL_TRUE, 0, array_size, &force[0],
                                 NULL, &event);
  err = queue.enqueueWriteBuffer(cl_vel, CL_TRUE, 0, array_size, &vel[0],
                                 NULL, &event);
  err = queue.enqueueWriteBuffer(cl_pos_gen, CL_TRUE, 0, array_size, &pos[0],
                                 NULL, &event);
  queue.finish();
}

void MD::popCorn() {
  printf("in popCorn\n");
  // Initialize our kernel from the program.
  try {
    forceKernel = cl::Kernel(program, "force", &err);
    updateKernel = cl::Kernel(program, "update", &err);
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
  }
  printf("here!\n");
  // Set the arguements of our kernel.
  try {
    err = forceKernel.setArg(0, cl_vbos[0]); // Position vbo.
    err = forceKernel.setArg(1, cl_vbos[1]); // Color vbo.
    err = forceKernel.setArg(2, cl_forces);
    err = forceKernel.setArg(3, cl_pos_gen);
    err = updateKernel.setArg(0, cl_vbos[0]); // Position vbo.
    err = updateKernel.setArg(1, cl_vbos[1]); // Color vbo.
    err = updateKernel.setArg(2, cl_forces);
    err = updateKernel.setArg(3, cl_vel);
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
  }
  printf("there!\n");
  // Wait for the command queue to finish these commands before proceeding.
  queue.finish();
  printf("Done popCorn\n");
}

void MD::runKernel() {
  // This will update our system by calculating new velocity and updating the
  // positions of our particles.
  // Make sure OpenGL is done using our VBOs.
  glFinish();
  // Map OpenGL buffer object for writing from OpenCL.
  // This passes in the vector of VBO buffer objects (position and color).
  err = queue.enqueueAcquireGLObjects(&cl_vbos, NULL, &event);
  //printf("acquire: %s\n", oclErrorString(err));
  queue.finish();

  float dt = .01f;
  forceKernel.setArg(4, num);  //pass in the timestep
  updateKernel.setArg(4, dt);  //pass in the timestep
  // Execute the kernel.
  err = queue.enqueueNDRangeKernel(forceKernel, cl::NullRange, cl::NDRange(num),
    cl::NullRange, NULL, &event);
  err = queue.enqueueNDRangeKernel(updateKernel, cl::NullRange,
                                   cl::NDRange(num), cl::NullRange, NULL,
                                   &event);
  //printf("clEnqueueNDRangeKernel: %s\n", oclErrorString(err));

  queue.finish();

  // Release the VBOs so OpenGL can play with them.
  err = queue.enqueueReleaseGLObjects(&cl_vbos, NULL, &event);
  //printf("release gl: %s\n", oclErrorString(err));
  /*cl_float3 A[num];
  err = queue.enqueueReadBuffer(cl_vbos[0], CL_TRUE, 0, array_size, &A[0],
  NULL, &event);*/
  queue.finish();
  /*for (int i = 0; i < 100; i++)
    std::cout << A[i] << " ";
    std::cout << std::endl;*/
}
