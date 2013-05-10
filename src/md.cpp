#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>


// Needed for context sharing functions.
#include <GL/glx.h>


// Local includes.
#include "md.hpp"
#include "util.hpp"
#include "types.hpp"


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

  // This part of the setup may be Linux specific.
  cl_context_properties props[] =
    {
      CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
      CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(),
      0
    };
  try {
    context = cl::Context(CL_DEVICE_TYPE_GPU, props);
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    exit(EXIT_FAILURE);
  }

  // Create the command queue we will use to execute OpenCL commands.
  try {
    queue = cl::CommandQueue(context, devices[deviceUsed], 0, &err);
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%d)\n", er.what(), er.err());
    exit(EXIT_FAILURE);
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


void MD::loadProgram(std::string kernel_source, int group_size_val) {
  // Program Setup.
  int pl;
  group_size = group_size_val;
  printf("Load the program.\n");
  bool failed = false;

  pl = kernel_source.size();
  printf("Kernel size: %d.\n", pl);
  try {
    cl::Program::Sources source(1,
                                std::make_pair(kernel_source.c_str(), pl));
    program = cl::Program(context, source);
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
  }

  printf("Building program...\n");
  try {
    //err = program.build(devices, "-cl-nv-verbose -cl-nv-maxrregcount=100");
    std::stringstream build_options;
    // Define the group size to allow for __local arrays.
    build_options << "-D SIZE=" << group_size << std::ends;
    err = program.build(devices, build_options.str().c_str());
  }
  catch (cl::Error er) {
    printf("program.build: %s\n", oclErrorString(er.err()));
    failed = true;
  }
  printf("Done building program.\n");
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
  printf("Set up GL sharing.\n");
  try {
    // Create OpenCL buffer from GL VBO.
    // We don't need to push any data here because it's already in the VBO.
    cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_WRITE, pos_vbo, &err));
    cl_vbos.push_back(cl::BufferGL(context, CL_MEM_READ_WRITE, col_vbo, &err));
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    exit(EXIT_FAILURE);
  }

  // Create the OpenCL only arrays.
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
  queue.finish();
}


void MD::clInit(float bound, float dt, std::string force_kernel_name) {
  printf("Initializing CL Kernels.\n");
  // Initialize our kernel from the program.
  try {
    forceKernel = cl::Kernel(program, force_kernel_name.c_str(), &err);
    updateKernel = cl::Kernel(program, "update", &err);
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    exit(EXIT_FAILURE);
  }
  // Set the arguements of our kernel.
  try {
    err = forceKernel.setArg(0, cl_vbos[0]);  // Position vbo.
    err = forceKernel.setArg(1, cl_vbos[1]);  // Color vbo.
    err = forceKernel.setArg(2, cl_forces);
    err = forceKernel.setArg(3, num);          // Pass in the size.
    err = updateKernel.setArg(0, cl_vbos[0]);  // Position vbo.
    err = updateKernel.setArg(1, cl_vbos[1]);  // Color vbo.
    err = updateKernel.setArg(2, cl_forces);
    err = updateKernel.setArg(3, cl_vel);
    err = updateKernel.setArg(4, bound);
    updateKernel.setArg(5, dt);                // Pass in the timestep.
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
  }
  // Wait for the command queue to finish these commands before proceeding.
  queue.finish();
  printf("Done initializing.\n");
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

  // Execute the kernel.
  try {
    err = queue.enqueueNDRangeKernel(forceKernel, cl::NullRange,
                                     cl::NDRange(num),
                                     cl::NDRange(group_size), NULL, &event);
    //err = queue.enqueueNDRangeKernel(forceKernel, cl::NullRange,
    //                                 cl::NDRange(num),
    //                                   cl::NullRange, NULL, &event);
    err = queue.enqueueNDRangeKernel(updateKernel, cl::NullRange,
                                     cl::NDRange(num), cl::NullRange, NULL,
                                     &event);
    err = queue.finish();
  }
  catch (cl::Error er) {
    printf("ERROR: %s(%s)\n", er.what(), oclErrorString(er.err()));
    if (er.err() == -54)
      std::cout << "The group size must evenly divide the number of particles."
                << std::endl;
    exit(EXIT_FAILURE);
  }

  // Release the VBOs so OpenGL can play with them.
  err = queue.enqueueReleaseGLObjects(&cl_vbos, NULL, &event);
  queue.finish();
}
