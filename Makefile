
CXX = g++
CXXFLAGS = -g -std=gnu++17 -DOMPI_SKIP_MPICXX
LIBRARIES = -lpagmo -lboost_serialization -ltbb -pthread -lR -lRInside 
INCLUDES = -I /home/amoongat/.local/include -I"/home/amoongat/R/lib64/R/include" -I"/home/amoongat/R-4.3.1/library/Rcpp/include" -I"/home/amoongat/R-4.3.1/library/RcppArmadillo/include" -I"/home/amoongat/R-4.3.1/library/RInside/include" -I"/usr/include/python3.10" -I"/usr/lib/python3/dist-packages/numpy/core/include" -I"/home/amoongat/pagmo/boost-ver/include"
LIB_PATHS = -L /home/amoongat/.local/lib64 -L/home/amoongat/R-4.3.1/lib -L"/home/amoongat/R-4.3.1/library/RInside/lib" -L"/home/amoongat/pagmo/boost-ver/lib" -L"/home/amoongat/pagmo/oneTBB/my_installed_onetbb/lib64" 
RPATH = -Wl,-R/home/amoongat/.local/lib64 -Wl,-rpath,/home/amoongat/R-4.3.1/lib -Wl,-rpath,/home/amoongat/R-4.3.1/library/RInside/lib -Wl,-rpath,/home/amoongat/pagmo/boost-ver/lib -Wl,-rpath,/home/amoongat/pagmo/oneTBB/my_installed_onetbb/lib64

SRC =  InverseRL.cpp Strategy.cpp aca2.cpp discountedRwdQlearning.cpp avgRewardQLearning.cpp Pagmoprob.cpp PagmoMle.cpp PagmoMultiObjCluster.cpp InferStrategy.cpp Simulation.cpp runEM.cpp
TARGET = inverseRL

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(INCLUDES) $(LIB_PATHS) $(LIBRARIES) $(RPATH)

.PHONY: clean
clean:
	rm -f $(TARGET)

