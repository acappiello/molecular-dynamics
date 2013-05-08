// createVBO routine and my first experience with VBOs from:
// http://www.songho.ca/opengl/gl_vbo.html

#ifndef ADVCL_UTIL_H_INCLUDED
#define ADVCL_UTIL_H_INCLUDED


#include <CL/cl_platform.h>


const char* oclErrorString(cl_int error);


// Create a VBO.
// Target is usually GL_ARRAY_BUFFER.
// Usage is usually GL_DYNAMIC_DRAW.
GLuint createVBO(const void* data, int dataSize, GLenum target, GLenum usage);


static inline cl_float3 f3(float x, float y, float z) {
  cl_float4 a;
  a.x = x;
  a.y = y;
  a.z = z;
  return a;
}


static inline cl_float4 f4(float x, float y, float z, float w) {
  cl_float4 a;
  a.x = x;
  a.y = y;
  a.z = z;
  a.w = w;
  return a;
}

#endif
