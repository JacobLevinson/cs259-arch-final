CXX		=	g++
CXXFLAGS	=	-g -O3 -Wall -fopenmp

all:		predict

predict:	predict.cc trace.cc predictor.h branch.h trace.h my_predictor_gpu.h
		$(CXX) $(CXXFLAGS) -o predict predict.cc trace.cc

clean:
		rm -f predict
