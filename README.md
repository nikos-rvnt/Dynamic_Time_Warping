# Dynamic_Time_Warping
python, c++ and cuda implementation of Dynamic Time Warping Distance algorithm

Libraries needed: 
  - Python: Numpy (https://numpy.org/install/)
  - C++: Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page), JSON for Modern C++ (https://github.com/nlohmann/json)
 
 To compile:
  - C++: g++ test_detewe.cpp -o test_detewe -lstdc++fs -std=c++17
  - CUDA: nvcc -g -G  -lstdc++fs -std=c++17 -rdc=true -lcudadevrt --expt-relaxed-constexpr test_dtwGPU.cu -o testDTW
