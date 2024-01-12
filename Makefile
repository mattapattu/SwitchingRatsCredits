
CXX = g++
CXXFLAGS = -g -std=gnu++17
LIBRARIES = -lpagmo -lboost_serialization -ltbb -pthread -lR -lRInside 
INCLUDES = -I /home/amoongat/.local/include -I"/usr/include/R" -I"/home/amoongat/R/x86_64-redhat-linux-gnu-library/4.0/Rcpp/include" -I"/home/amoongat/R/x86_64-redhat-linux-gnu-library/4.0/RcppArmadillo/include" -I"/home/amoongat/R/x86_64-redhat-linux-gnu-library/4.0/RInside/include" -I"/usr/include/python3.10" -I"/usr/lib/python3/dist-packages/numpy/core/include" -I"/home/amoongat/pagmo/boost-ver/include"
LIB_PATHS = -L /home/amoongat/.local/lib64 -L/usr/lib64/R/lib -L"/home/amoongat/R/x86_64-redhat-linux-gnu-library/4.0/RInside/lib" -L"/home/amoongat/pagmo/boost-ver/lib" -L"/tmp/my_installed_onetbb/lib64"
RPATH = -Wl,-R/home/amoongat/.local/lib64 -Wl,-rpath,/home/amoongat/R/x86_64-redhat-linux-gnu-library/4.0/Rcpp/RInside/lib

SRC =  InverseRL.cpp Strategy.cpp aca2.cpp discountedRwdQlearning.cpp avgRewardQLearning.cpp Pagmoprob.cpp PagmoMle.cpp PagmoMultiObjCluster.cpp InferStrategy.cpp Simulation.cpp runEM.cpp
TARGET = inverseRL

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(INCLUDES) $(LIB_PATHS) $(LIBRARIES) $(RPATH)

.PHONY: clean
clean:
	rm -f $(TARGET)

