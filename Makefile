
EXECUTABLE := md

FILES      := main md util

CL_FILES   := md.cl

###########################################################

CXX=g++
CXXFLAGS=-Wall -g #-DCL_USE_DEPRECATED_OPENCL_1_1_APIS

ARCH=$(shell uname | sed -e 's/-.*//g')
SRCDIR=src
OBJDIR=objs

INCL       := ./opencl11

INCLUDES   := $(addprefix -I, $(INCL))

LIBS       := GL glut OpenCL GLU GLEW
LDFLAGS=

LDLIBS     := $(addprefix -l, $(LIBS))

CC_FILES   := $(addsuffix .cpp, $(FILES))
OB         := $(addsuffix .o, $(FILES))
OBJS       := $(addprefix $(OBJDIR)/, $(OB))

.PHONY: dirs clean $(SRCDIR)/main.hpp

default: $(EXECUTABLE)

dirs:
	mkdir -p $(OBJDIR)/

clean:
	rm -rf $(OBJDIR) *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/%.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -c -o $@
