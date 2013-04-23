#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <math.h>


// OpenGL stuff.
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <GL/glext.h>

// Our OpenCL Particle Systemclass.
#include "md.hpp"
#include "cycle_timer.hpp"

#define NUM_PARTICLES 20000

static struct prog_state {
  MD *md;
  // GL related variables.
  int window_width;
  int window_height;
  int glutWindowHandle;
  float translate_z;
  int mouse_old_x, mouse_old_y;
  int mouse_buttons;
  float rotate_x, rotate_y;
  double t0;
  double tlast;
  int frames;
  int framerate;
} prog_state;

// Main app helper functions.
void init_gl(int argc, char** argv);
void appRender();
void appDestroy();
void timerCB(int ms);
void appKeyboard(unsigned char key, int x, int y);
void appMouse(int button, int state, int x, int y);
void appMotion(int x, int y);

// Quick random function to distribute our initial points.
float rand_float(float mn, float mx) {
  float r = random() / (float) RAND_MAX;
  return mn + (mx-mn)*r;
}

int main(int argc, char** argv) {
  prog_state.window_width = 800;
  prog_state.window_height = 600;
  prog_state.glutWindowHandle = 0;
  prog_state.translate_z = -1.f;
  prog_state.mouse_buttons = 0;
  prog_state.rotate_x = 0.f;
  prog_state.rotate_y = 0.f;
  prog_state.t0 = CycleTimer::currentSeconds();
  prog_state.tlast = 0.f;
  prog_state.frames = 0;
  prog_state.framerate = 0;
  // Setup our GLUT window and OpenGL related things.
  // Glut callback functions are setup here too.
  init_gl(argc, argv);

  // Initialize our MD object, this sets up the context.
  prog_state.md = new MD();

  // Load and build our CL program from the file.
  //#include "md.cl"  //std::string kernel_source is defined in this file.
  std::string kernel_source = prog_state.md->loadFile("src/md.cl");
  prog_state.md->loadProgram(kernel_source);

  // Initialize our particle system with positions, velocities and color.
  int num = NUM_PARTICLES;
  std::vector<Vec4> pos(num);
  std::vector<Vec4> vel(num);
  std::vector<Vec4> color(num);

  // Fill our vectors with initial data.
  for(int i = 0; i < num; i++) {
    // Distribute the particles in a random circle around z axis.
    float rad = rand_float(.2, .5);
    float x = rad*sin(2*3.14 * i/num);
    float z = 0.0f;  // -.1 + .2f * i/num;
    float y = rad*cos(2*3.14 * i/num);
    pos[i] = Vec4(x, y, z, 1.0f);

    // Give some initial velocity.
    //float xr = rand_float(-.1, .1);
    //float yr = rand_float(1.f, 3.f);
    // The life is the lifetime of the particle: 1 = alive 0 = dead.
    // As you will see in part2.cl we reset the particle when it dies.
    float life_r = rand_float(0.f, 1.f);
    vel[i] = Vec4(0.f, 0.f, 3.0f, life_r);

    // Just make them red and full alpha.
    color[i] = Vec4(1.0f, 0.0f, 0.0f, 1.0f);
  }
  std::cout << "foo: " << &pos[0] << " bar: " << &pos[num] << std::endl;
  std::cout << "num: " << num << " sizeof(Vec4): " << sizeof(Vec4) << std::endl;

  prog_state.md->loadData(pos, vel, color);

  prog_state.md->popCorn();

  // This starts the GLUT program, from here on out everything we want
  // to do needs to be done in glut callback functions.
  glutMainLoop();

  return EXIT_SUCCESS;
}

void appRender() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // This updates the particle system by calling the kernel.
  prog_state.md->runKernel();

  prog_state.frames++;
  double tnow = CycleTimer::currentSeconds() - prog_state.t0;
  if (tnow > prog_state.tlast + 1.f) {
    prog_state.framerate = prog_state.frames;
    //std::cout << "Frames: " << prog_state.framerate << std::endl;
    prog_state.tlast = tnow;
    prog_state.frames = 0;
  }

  // Render the particles from VBOs.
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_POINT_SMOOTH);
  glPointSize(5.);

  //printf("color buffer\n");
  glBindBuffer(GL_ARRAY_BUFFER, prog_state.md->c_vbo);
  glColorPointer(4, GL_FLOAT, 0, 0);

  //printf("vertex buffer\n");
  glBindBuffer(GL_ARRAY_BUFFER, prog_state.md->p_vbo);
  glVertexPointer(4, GL_FLOAT, 0, 0);

  //printf("enable client state\n");
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  // Need to disable these for blender.
  glDisableClientState(GL_NORMAL_ARRAY);

  //printf("draw arrays\n");
  glDrawArrays(GL_POINTS, 0, prog_state.md->num);

  //printf("disable stuff\n");
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  // Display framerate.
  char buf[10];
  sprintf(buf, "%d", prog_state.framerate);
  std::string msg = std::string(buf);
  glRasterPos2f(-1.3f, 0.9f);
  glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
  glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)msg.c_str());

  glutSwapBuffers();
}

void init_gl(int argc, char** argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glutInitWindowSize(prog_state.window_width, prog_state.window_height);
  glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 -
                          prog_state.window_width/2,
                          glutGet(GLUT_SCREEN_HEIGHT)/2 -
                          prog_state.window_height/2);


  std::stringstream ss;
  ss << "Adventures in OpenCL: Part 2, " << NUM_PARTICLES << " particles" <<
    std::ends;
  prog_state.glutWindowHandle = glutCreateWindow(ss.str().c_str());

  glutDisplayFunc(appRender);      // Main rendering function.
  glutTimerFunc(30, timerCB, 30);  // Determin a minimum time between frames.
  glutKeyboardFunc(appKeyboard);
  glutMouseFunc(appMouse);
  glutMotionFunc(appMotion);

  glewInit();

  glClearColor(0.0, 0.0, 0.0, 1.0);
  glDisable(GL_DEPTH_TEST);

  // Viewport.
  glViewport(0, 0, prog_state.window_width, prog_state.window_height);

  // Projection.
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(90.0, (GLfloat)prog_state.window_width /
                 (GLfloat)prog_state.window_height, 0.1,
                 1000.0);

  // Set view matrix.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, prog_state.translate_z);
}

void appDestroy() {
  // This makes sure we properly cleanup our OpenCL context.
  //delete example;
  if (prog_state.glutWindowHandle) glutDestroyWindow(prog_state.glutWindowHandle);
  printf("about to exit!\n");

  glutLeaveMainLoop();
  std::cout << "^^" << std::endl;
  exit(EXIT_SUCCESS);
}

void timerCB(int ms) {
  // This makes sure the appRender function is called every ms miliseconds.
  glutTimerFunc(ms, timerCB, ms);
  glutPostRedisplay();
}

void appKeyboard(unsigned char key, int x, int y) {
  // This way we can exit the program cleanly.
  switch(key) {
  case '\033': // escape quits
  case '\015': // Enter quits
  case 'Q':    // Q quits
  case 'q':    // q (or escape) quits
    // Cleanup up and quit
    //appDestroy();
    exit(0);
    break;
  }
}

void appMouse(int button, int state, int x, int y) {
  // Handle mouse interaction for rotating/zooming the view.
  if (state == GLUT_DOWN) {
    prog_state.mouse_buttons |= 1<<button;
  } else if (state == GLUT_UP) {
    prog_state.mouse_buttons = 0;
  }

  prog_state.mouse_old_x = x;
  prog_state.mouse_old_y = y;
}

void appMotion(int x, int y) {
  // Handle the mouse motion for zooming and rotating the view.
  float dx, dy;
  dx = x - prog_state.mouse_old_x;
  dy = y - prog_state.mouse_old_y;

  if (prog_state.mouse_buttons & 1) {
    prog_state.rotate_x += dy * 0.2;
    prog_state.rotate_y += dx * 0.2;
  } else if (prog_state.mouse_buttons & 4) {
    prog_state.translate_z += dy * 0.1;
  }

  prog_state.mouse_old_x = x;
  prog_state.mouse_old_y = y;

  // Set view matrix.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(0.0, 0.0, prog_state.translate_z);
  glRotatef(prog_state.rotate_x, 1.0, 0.0, 0.0);
  glRotatef(prog_state.rotate_y, 0.0, 1.0, 0.0);
}
