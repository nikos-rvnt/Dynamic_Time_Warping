# Dynamic_Time_Warping
python, c++ and cuda implementation of Dynamic Time Warping Distance algorithm

Libraries needed: 
  - Python: Numpy (https://numpy.org/install/)
  - C++: Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page), JSON for Modern C++ (https://github.com/nlohmann/json)
 
 To compile:
  - C++: g++ test_detewe.cpp -o test_detewe -lstdc++fs -std=c++17
  - CUDA: nvcc -g -G  -lstdc++fs -std=c++17 -rdc=true -lcudadevrt --expt-relaxed-constexpr test_dtwGPU.cu -o testDTW

## Time comparison results 
DTW was computed for two equal length (128) sequences.

- Run Time on Python (+Numpy):
      6.636e-02 sec

- Run Time on C++ (+Eigen):
      7.476e-03 sec

- Run Time on CUDA (Nvidia Titan Xp):
      1.036e-05 sec

