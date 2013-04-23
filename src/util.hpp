// createVBO routine and my first experience with VBOs from:
// http://www.songho.ca/opengl/gl_vbo.html

#ifndef ADVCL_UTIL_H_INCLUDED
#define ADVCL_UTIL_H_INCLUDED

const char* oclErrorString(cl_int error);

// Create a VBO.
// Target is usually GL_ARRAY_BUFFER.
// Usage is usually GL_DYNAMIC_DRAW.
GLuint createVBO(const void* data, int dataSize, GLenum target, GLenum usage);

#endif
