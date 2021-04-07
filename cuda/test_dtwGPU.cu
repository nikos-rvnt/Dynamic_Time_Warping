
//nvcc -g -G  -lstdc++fs -std=c++17 -rdc=true -lcudadevrt --expt-relaxed-constexpr test_dtwGPU.cu -o testDTW
#include <fstream>
#include <nlohmann/json.hpp>

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <limits>

#include <eigen3/Eigen/Dense>

//#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#include "detewe.cpp"
#include "deteweGPU.cuh"
//#include "lower_boundsGPU.cuh"
#include "splineInterpolation.h"

namespace fs = std::experimental::filesystem;

using json = nlohmann::json;
using namespace std;
using namespace Eigen;
using namespace chrono;



void eigenMatrixData2CUDA(const float* eigMatCol, int matRows, int matCols, float *cudaVec)
{
	//float* matBack;
	const int len = matRows*matCols;
	const int sizeMatBack = sizeof(float)*len;
	
	cudaMalloc((void**)&cudaVec, sizeMatBack);
	cudaMemcpy( cudaVec, eigMatCol, sizeMatBack, cudaMemcpyHostToDevice);
	
	//return matBack;
}


int main()
{

	string pathAnno = "/media/nikos/Data/NAS_healthPose/Athens 2019-07-23/HealthSign_ximea/24_07/";
	vector<string> pathGlossAnno;
	for (const auto & entry : fs::directory_iterator(pathAnno))
        pathGlossAnno.push_back(entry.path());

    //cout << pathGlossAnno[0] << endl;
    ifstream glossAnno1(pathGlossAnno[0]);
	json j1;
	j1 = json::parse(glossAnno1);

    int i, j, i_temp;
    int w, version = 0, symmetry = 1;

    double x4, y4;
    MatrixXf search(128,2), query(128,2);
    
    i_temp = 0;
    for( i=48; i<176; i++)
    {
    	// extract 4th point (x,y) 
		x4 = (double)j1["joints_2d_right"][i][6];
		y4 = (double)j1["joints_2d_right"][i][7];    	

		query(i_temp,0) = x4;
		query(i_temp,1) = y4;
		i_temp++;
    }

    i_temp = 0;
    for( i=88; i<216; i++)
    {
		x4 = (double)j1["joints_2d_right"][i][6];
		y4 = (double)j1["joints_2d_right"][i][7];    	

		search(i_temp,0) = x4;
		search(i_temp,1) = y4;
    	i_temp++;
    }

    //w = abs( search->rows() - query->rows()) + 25;
    w = (int)(0.2*query.rows());


    // cuda vars
    float *d_distance, *distDTW;
    //vector<vector<int>> d_path;
    float *d_search, *d_query ;
    float *queryArr = query.data();
    float *searchArr = search.data();
    int rowsQ = query.rows(), rowsS = search.rows();
    int feats = query.cols();
    float *cost_mat, *dtw_mat;


    float** costMat = new float*[rowsQ];
    costMat[0] = new float[rowsQ * rowsS];
    for ( i = 1; i < rowsQ; ++i) costMat[i] = (costMat[i-1] + rowsS);

    for ( i = 0; i < rowsQ; ++i) {
        for ( j = 0; j < rowsS; ++j) {
            costMat[i][j] = i*rowsS+j;;
        }
    }
    float** dtwMat = new float*[rowsQ];
    dtwMat[0] = new float[rowsQ * rowsS];
    for ( i = 1; i < rowsQ; ++i) dtwMat[i] = (dtwMat[i-1] + rowsS);

    for ( i = 0; i < rowsQ; ++i) {
        for ( j = 0; j < rowsS; ++j) {
            dtwMat[i][j] = i*rowsS+j;
        }
    }


    size_t size_float = sizeof(float);
    //int size_vec = sizeof(vector<vector<int>>);
    size_t size_Mat_search = sizeof(float)*search.rows()*search.cols();
    size_t size_Mat_query = sizeof(float)*query.rows()*query.cols();


    // memory allocation
    distDTW = new float[2];
    cudaMalloc((void**)&d_distance, size_float*2);
	cudaMalloc((void**)&d_search, size_Mat_search);
	cudaMalloc((void**)&d_query, size_Mat_query);
    //cudaMalloc((void**)&d_path, size_vec);  --> des to segmentation fault (core dumped)
    size_t pitch1, pitch2;
    cudaMallocPitch((void**)&dtw_mat, &pitch1, rowsS, rowsQ);
    cudaMallocPitch((void**)&cost_mat, &pitch2, rowsS, rowsQ);
    
    // Copy inputs to device
    cudaMemcpy2D( dtw_mat, pitch1, dtwMat, rowsS*sizeof(float), rowsS*sizeof(float), rowsQ*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy2D( cost_mat, pitch2, costMat, rowsS*sizeof(float), rowsS*sizeof(float), rowsQ*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_search, searchArr, size_Mat_search, cudaMemcpyHostToDevice);
	cudaMemcpy( d_query, queryArr, size_Mat_query, cudaMemcpyHostToDevice);

    // unsigned int blocksize = 1;  
    // unsigned int nblocks = 1;
    // int blocksize = 128;
    // int nblocks = (int)(rowsQ + blocksize - 1)/blocksize;
    dim3 threads_per_block( 1, 256, 1 );
    dim3 blocks_in_grid( 1, ceil( float(rowsQ) / threads_per_block.y ), 1 );

    auto sum = std::chrono::duration<double>(0.0);
    for( i=0; i<10; i++)
    {
        steady_clock sc;    // time counter (sec)
        auto start_timer = sc.now();     

        //DeTeWe_GPU<<<nblocks,blocksize>>>( d_query, d_search, rowsQ, rowsS, feats, w, version, symmetry, d_distance, cost_mat, dtw_mat );
        DeTeWe_GPU<<<blocks_in_grid,threads_per_block>>>( d_query, d_search, rowsQ, rowsS, feats, w, version, symmetry, d_distance, cost_mat, dtw_mat );
      	auto end_timer = sc.now();
        auto time_diff = end_timer - start_timer; 
        sum += static_cast<duration<double>>(time_diff);
      	
        cout << "Time on GPU: " << static_cast<duration<double>>(end_timer - start_timer).count() << '\n';
    }
    cout << "Mean time: " << sum / static_cast<duration<double>>(10) ;

    // tell host to wait until device has finished
   	cudaDeviceSynchronize();
	// cudaError_t error = cudaGetLastError();
	// if(error!=cudaSuccess)
	// {
 //   		fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
 //   		exit(-1);
	// }
    

    // Copy result back to host
	// cudaMemcpy(&path, d_path.data(), size_vec, cudaMemcpyDeviceToHost);
	cudaMemcpy( distDTW, d_distance, size_float, cudaMemcpyDeviceToHost);
    printf("DTW__Dist: %f \n\n", distDTW[0]);

	// Cleanup
    free(searchArr); free(queryArr);

	// for(int i = 0; i < rowsQ; i++) delete[] costMat[i];
 //    delete[] costMat;
    
 //    for(int i = 0; i < rowsQ; i++) delete[] dtwMat[i];
 //    delete[] dtwMat;

    delete[] distDTW;

    cudaFree(cost_mat); cudaFree(dtw_mat);
    cudaFree(d_search); cudaFree(d_query); cudaFree(d_distance); cudaFree(cost_mat), cudaFree(dtw_mat); 
	

    return 0;

}

