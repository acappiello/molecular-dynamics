Molecular Dynamics
==================

General
-------

My final project from 15-418, Parallel Computer Architecture and Programming, at CMU.

For more, see http://www.alexcappiello.com/projects/15418/.

I can only guarantee it will compile and run on linux with an NVIDIA OpenCL device because that's all I've tried it on. Minor changes need to be made to use with a CPU implementation of OpenCL.

Dependencies
------------

* OpenCL
* OpenGL (mesa), freeglut3, and glew

Compiling and Running
---------------------

```
$ make
```

```
$ ./md -?
Usage: ./md [options]
Program Options:
  -w  --width <INT>         Window Width               default=800
  -h  --height <INT>        Window Height              default=800
  -n  --nparticles <INT>    Number of particles        default=1024
  -b  --bbox <FLOAT>        Size of bounding box (+-)  default=50.000000
  -g  --group-size <INT>    Size of the local group    default=32
  -t  --dt <FLOAT>          Time step                  default=1.000000e-15
  -k  --force-kernel <STR>  Force kernel to use        default=force_naive
  -?  --help                This message
```
