#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <time.h>


// OpenGL stuff.
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <GL/glext.h>


// Other local includes.
#include "md.hpp"
#include "cycle_timer.hpp"
#include "util.hpp"


static struct prog_state {
  // Class instance.
  MD *md;
  // GL related variables.
  int window_width;
  int window_height;
  int glutWindowHandle;
  float translate_z;
  int mouse_old_x, mouse_old_y;
  int mouse_buttons;
  float rotate_x, rotate_y;
  // Internal variables and parameters.
  double t0;
  double tlast;
  int frames;
  int framerate;
  size_t nparticles;
  float bbox;
  int group_size;
  float dt;
  std::string force_kernel_name;
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


void set_default_state () {
  prog_state.window_width = 800;
  prog_state.window_height = 800;
  prog_state.glutWindowHandle = 0;
  prog_state.mouse_buttons = 0;
  prog_state.rotate_x = 0.f;
  prog_state.rotate_y = 0.f;
  prog_state.t0 = CycleTimer::currentSeconds();
  prog_state.tlast = 0.f;
  prog_state.frames = 0;
  prog_state.framerate = 0;
  prog_state.nparticles = 1024;
  prog_state.bbox = 50.f;
  prog_state.translate_z = -2.1f * prog_state.bbox;
  prog_state.group_size = 32;
  prog_state.dt = 1e-15;
  prog_state.force_kernel_name = std::string("force_naive");
}


void usage(const char* progname) {
  set_default_state();
  printf("Usage: %s [options]\n", progname);
  printf("Program Options:\n");
  printf("  -w  --width <INT>         Window Width               default=%d\n",
         prog_state.window_width);
  printf("  -h  --height <INT>        Window Height              default=%d\n",
         prog_state.window_height);
  printf("  -n  --nparticles <INT>    Number of particles        default=%zu\n",
         prog_state.nparticles);
  printf("  -b  --bbox <FLOAT>        Size of bounding box (+-)  default=%f\n",
         prog_state.bbox);
  printf("  -g  --group-size <INT>    Size of the local group    default=%d\n",
         prog_state.group_size);
  printf("  -t  --dt <FLOAT>          Time step                  default=%e\n",
         prog_state.dt);
  printf("  -k  --force-kernel <STR>  Force kernel to use        default=%s\n",
         prog_state.force_kernel_name.c_str());
  printf("  -?  --help                This message\n");
}


int main(int argc, char** argv) {
  set_default_state();
  srandom(time(NULL));

  int opt;
  static struct option long_options[] = {
    {"help",     0, 0,  '?'},
    {"width",    0, 0,  'w'},
    {"height",   0, 0,  'h'},
    {"nparticles",     0, 0,  'n'},
    {"bbox",      0, 0,  'b'},
    {"group-size",     0, 0,  'g'},
    {"dt",       0, 0,  't'},
    {"force-kernel",  0, 0,   'k'},
    {0 ,0, 0, 0}
  };

  while ((opt = getopt_long(argc, argv, "w:h:n:b:g:t:k:?", long_options, NULL))
         != EOF) {
    switch (opt) {
    case 'w':
      prog_state.window_width = atoi(optarg);
      break;
    case 'h':
      prog_state.window_height = atoi(optarg);
      break;
    case 'n':
      prog_state.nparticles = atoi(optarg);
      break;
    case 'b':
      prog_state.bbox = atof(optarg);
      prog_state.translate_z = -2.2f * prog_state.bbox;
      break;
    case 'g':
      prog_state.group_size = atoi(optarg);
      break;
    case 't':
      prog_state.dt = atof(optarg);
      break;
    case 'k':
      prog_state.force_kernel_name = std::string(optarg);
      break;
    case '?':
    default:
      usage(argv[0]);
      return 1;
    }
  }

  if (prog_state.nparticles % prog_state.group_size != 0) {
    std::cout
      << "ERROR: The group size must evenly divide the number of particles."
      << std::endl;
    exit(EXIT_FAILURE);
  }

  // Setup our GLUT window and OpenGL related things.
  // Glut callback functions are setup here too.
  init_gl(argc, argv);

  // Initialize our MD object, this sets up the context.
  prog_state.md = new MD();

  // Load and build our CL program from the file.
  // Presently, this means that you can't run the program from another dir.
  std::string kernel_source = prog_state.md->loadFile("src/md.cl");
  prog_state.md->loadProgram(kernel_source, prog_state.group_size);

  // Initialize the particle system with positions, velocities and color.
  int num = prog_state.nparticles;
  std::vector<cl_float4> pos(num);
  std::vector<cl_float4> force(num);
  std::vector<cl_float4> vel(num);
  std::vector<cl_float4> color(num);

  // Fill the vectors with initial data.
  for(int i = 0; i < num; i++) {
    // Distribute the particles in a random cube +- bbox in all directions.
    float max = prog_state.bbox;
    float min = -1.f * max;
    float x = rand_float(min, max);
    float z = rand_float(min, max);
    float y = rand_float(min, max);
    pos[i] = f4(x, y, z, 1.f);

    // Give some initial velocity. Otherwise, things are boring initially.
    max /= 10;
    min /= 10;
    x = rand_float(min, max);
    z = rand_float(min, max);
    y = rand_float(min, max);
    vel[i] = f4(x, y, z, 0.f);

    force[i] = f4(0.f, 0.f, 0.f, 0.f);

    // Just make them red and full alpha now. The kernel will reassign colors.
    color[i] = f4(1.0f, 0.0f, 0.0f, 1.0f);
  }

  // Move this data to the CL device.
  prog_state.md->loadData(pos, force, vel, color);

  // Set up the kernel functions.
  prog_state.md->clInit(prog_state.bbox, prog_state.dt,
                        prog_state.force_kernel_name);

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

  // Color buffer.
  glBindBuffer(GL_ARRAY_BUFFER, prog_state.md->col_vbo);
  glColorPointer(4, GL_FLOAT, 0, 0);

  // Vertex buffer.
  glBindBuffer(GL_ARRAY_BUFFER, prog_state.md->pos_vbo);
  glVertexPointer(4, GL_FLOAT, 0, 0);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  glDisableClientState(GL_NORMAL_ARRAY);

  glDrawArrays(GL_POINTS, 0, prog_state.md->num);

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  // Display framerate.
  std::stringstream fr;
  fr << prog_state.framerate;
  glRasterPos3f(-prog_state.bbox, prog_state.bbox, prog_state.bbox);
  glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
  glutBitmapString(GLUT_BITMAP_HELVETICA_18,
                   (const unsigned char*)fr.str().c_str());

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

  // Give the window a descriptive title.
  std::stringstream ss;
  ss << "md:: nparticles: " << prog_state.nparticles << ", box size: "
     << prog_state.bbox << ", group size: " <<  prog_state.group_size
     << ", dt: " << prog_state.dt << ", kernel: "
     << prog_state.force_kernel_name << std::ends;
  prog_state.glutWindowHandle = glutCreateWindow(ss.str().c_str());

  // Set up callbacks.
  glutDisplayFunc(appRender);      // Main rendering function.
  glutTimerFunc(15, timerCB, 15);  // Determine a minimum time between frames.
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
  if (prog_state.glutWindowHandle)
    glutDestroyWindow(prog_state.glutWindowHandle);

  glutLeaveMainLoop();
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
