include ../../common/make.config

#Can be changed by `make TYPE=CPU`
TYPE = GPU

#Library
ifeq ($(TYPE),GPU)
	OPENCL_INC = $(NV_OPENCL_INC)
	OPENCL_LIB = $(NV_OPENCL_LIB)
else
	OPENCL_INC = $(INTEL_OPENCL_INC)
	OPENCL_LIB = $(INTEL_OPENCL_LIB)
endif

CXX = g++
DEBUGFLAGS = -g
# You will need to modify the following 2 lines
# by inserting the correct path to the OpenCL
# lib and inc directories.
CPPFLAGS = -I$(OPENCL_INC) -I../util -DTIMING -c $(DEBUGFLAGS)
LDFLAGS = -L$(OPENCL_LIB)

# link objects(binaries) together
a.out: main.o OpenCL.o timing.o
	$(CXX) $(LDFLAGS) main.o OpenCL.o timing.o \
		-o pathfinder.out \
		-lOpenCL

# compile main function file into object (binary)
main.o: main.cpp OpenCL.cpp kernels.cl ../util/timing.h
	$(CXX) $(CPPFLAGS) -o main.o main.cpp

OpenCL.o: OpenCL.cpp ../util/timing.h
	$(CXX) $(CPPFLAGS) -o OpenCL.o OpenCL.cpp

timing.o: ../util/timing.c ../util/timing.h
	$(CXX) -I../util -c -DTIMING $(DEBUGFLAGS) -o timing.o ../util/timing.c

clean:
	clear
	@rm -f pathfinder.out *.o *.out *~ *\#

run: a.out
	./pathfinder.out -c 100000 -r 100 -h 20 -p 0 -d 0 > result.txt
