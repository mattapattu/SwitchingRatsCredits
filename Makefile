
CXX = g++
CXXFLAGS = -g -std=gnu++17
LIBRARIES = -lpagmo -lboost_serialization -ltbb -pthread -lR -lRInside  -lpython3.10
INCLUDES = -I /home/mattapattu/.local/include -I"/usr/share/R/include" -I"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/Rcpp/include" -I"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RcppArmadillo/include" -I"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RInside/include" -I"/usr/include/python3.10" -I"/usr/lib/python3/dist-packages/numpy/core/include"
LIB_PATHS = -L /home/mattapattu/.local/lib -L/usr/lib/R/lib -L"/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RInside/lib"
RPATH = -Wl,-R/home/mattapattu/.local/lib -Wl,-rpath,/home/mattapattu/R/x86_64-pc-linux-gnu-library/4.1/RInside/lib

SRC = InferStrategy.cpp InverseRL.cpp Strategy.cpp aca2.cpp discountedRwdQlearning.cpp avgRewardQLearning.cpp PagmoMle.cpp
TARGET = inverseRL

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(INCLUDES) $(LIB_PATHS) $(LIBRARIES) $(RPATH)

.PHONY: clean
clean:
	rm -f $(TARGET)

