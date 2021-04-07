
//g++ test_detewe.cpp -o test_detewe -lstdc++fs -std=c++17
#include <fstream>
#include <nlohmann/json.hpp>

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <experimental/filesystem>

#include <eigen3/Eigen/Dense>

//#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
//#include "detewe.cpp"
//#include "deteweGPU.cuh"
#include "lower_boundsGPU.cuh"
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

    int i, j, i_temp, symm1 = 1;
    int w, version, symmetry;
    version = 0;
    //int *symmetry = &symm1;

    float lb_dist = 0.0;
    double x4, y4;
    MatrixXf search(34,2), query(34,2);
    
 //    MatrixXf *search = new MatrixXf(34,2);
	// MatrixXf *query = new MatrixXf(34,2);

    i_temp = 0;
    for( i=48; i<82; i++){

    	// extract 4th point (x,y) 
		x4 = (double)j1["joints_2d_right"][i][6];
		y4 = (double)j1["joints_2d_right"][i][7];    	

		//query->coeffRef(i_temp,0) = x4;
		//query->coeffRef(i_temp,1) = y4;
		query(i_temp,0) = x4;
		query(i_temp,1) = y4;
		i_temp++;
    }

    i_temp = 0;
    for( i=88; i<122; i++){

		x4 = (double)j1["joints_2d_right"][i][6];
		y4 = (double)j1["joints_2d_right"][i][7];    	

		// search->coeffRef(i_temp,0) = x4;
		// search->coeffRef(i_temp,1) = y4;
		search(i_temp,0) = x4;
		search(i_temp,1) = y4;
		//cout << search(i_temp,0) << " " << search(i_temp,1) << endl ;
    	i_temp++;
    }


    //w = abs( search->rows() - query->rows()) + 25;
    w = (int)(0.2*query.rows());
    //cout << "W: " << w << endl;

 //    int lobidi = 3;

 //    steady_clock sc;
 //    auto start_timer = sc.now();

 //    if(lobidi==0)
	// 	lb_dist = Kim_bounDemLow_2pnt( search, query);
 //    else if(lobidi==1)
	//     lb_dist = Yi_bounDemLow( search, query);
	// else if(lobidi==2 || lobidi==3 || lobidi==4)
	// {
	//     int lenQ, lenC, lenQ_cols, lenC_cols, lenQ_in = query.rows(), lenC_in = search.rows();
	//     MatrixXf search_int = MatrixXf::Zero( max(query.rows(),search.rows()), max(query.cols(),search.cols()));
	//     MatrixXf query_int = MatrixXf::Zero( max(query.rows(),search.rows()), max(query.cols(),search.cols()));

	//     if(lenQ_in>lenC_in)    
	//     {
	// 		splineInterp( search, lenQ_in, search_int);
	//         query_int = query;
	//     }
	//     else if(lenQ_in<lenC_in)
	//     {
	//         splineInterp( query, lenC_in, query_int);
	//     	search_int = search;
	//     }
	//     else
	//     {
	//     	query_int = query;
	//     	search_int = search;
	//     }
	//     MatrixXf uppQ = MatrixXf::Zero( max(query.rows(),search.rows()), max(query.cols(),search.cols()));
	//     MatrixXf lowQ = MatrixXf::Zero( max(query.rows(),search.rows()), max(query.cols(),search.cols()));  
 //    	if(lobidi==2)
	//     	lb_dist = Keogh_bounDemLow_SakoeChiba( search_int, query_int, w, uppQ, lowQ);
	// 	else if(lobidi==3)
	// 		lb_dist = Improved_bounDemLow( search_int, query_int, w, uppQ, lowQ);
	// 	else if(lobidi==4){
	// 		int v = 4;
	// 		float d = 5.0;
	// 		lb_dist = Enhanced_bounDemLow( search_int, query_int, w, uppQ, lowQ, v, d);
	// 	}
	// }

 //    MatrixXf uppQ = MatrixXf::Zero( max(query.rows(),search.rows()), max(query.cols(),search.cols()));
 //    MatrixXf lowQ = MatrixXf::Zero( max(query.rows(),search.rows()), max(query.cols(),search.cols()));  
	// if(lobidi==2)
 //    	lb_dist = Keogh_bounDemLow_SakoeChiba( search, query, w, uppQ, lowQ);
	// else if(lobidi==3)
	// 	lb_dist = Improved_bounDemLow( query, search, w, uppQ, lowQ);
 //    auto end_timer = sc.now();
	// cout << static_cast<duration<double>>(end_timer - start_timer).count() << '\n';
 //    cout << "LB_dist: " << lb_dist;
    
    vector<vector<int>> path;

    vector<vector<float> > v_search( 34, vector<float> (2, 0.0));  
    vector<vector<float> > v_query( 34, vector<float> (2, 0.0));  
    float* d_vquery;

    // for(i=0; i<34; i++)
    // {
    // 	for(j=0; j<2; j++)
    // 	{
    // 		v_search[i][j] = search->coeffRef(i,j);
    //  		v_query[i][j] = query->coeffRef(i,j);
    //  	}
    // }

    int* resultMax;
    // cublasHandle_t my_handle;
    // cublasStatus_t my_status = cublasCreate(&my_handle);
    // my_status = cublasIsamax( myhandle, 34*2, , 1, resultMax);


    // cuda vars
    int *d_w, *d_version, *d_symmetry;
    float *d_distance, *distDTW;
    vector<vector<int>> d_path;
    float *d_search, *d_query ;
    float *queryArr = query.data();
    float *searchArr = search.data();
    int *lenQ, *lenS, rowsQ = query.rows(), rowsS = search.rows();
    int *numFeats, feats = query.cols();
    // for(int i=0; i<34; i++)
    // {
    //     printf("%f \n\n", searchArr[i]);
    //     //trajC++;
    // }

    int size_float = sizeof(float);
    int size_int = sizeof(int);
    int size_vec = sizeof(vector<vector<int>>);
    // int size_Mat_search = sizeof(float)*search->rows()*search->cols();
    // int size_Mat_query = sizeof(float)*query->rows()*query->cols();
    int size_Mat_search = sizeof(float)*search.rows()*search.cols();
    int size_Mat_query = sizeof(float)*query.rows()*query.cols();

    // int size_Mat_search = sizeof(MatrixXf(search->rows(),search->cols()));
    // int size_Mat_query = sizeof(MatrixXf(query->rows(),query->cols()));

    // memory allocation
    //distDTW = (float*)malloc(sizeof(float));
    distDTW = new float[2];
    cudaMalloc((void**)&numFeats, size_int);
    cudaMalloc((void**)&lenQ, size_int);
    cudaMalloc((void**)&lenS, size_int);
    cudaMalloc((void**)&d_w, size_int);
    cudaMalloc((void**)&d_version, size_int);
    cudaMalloc((void**)&d_symmetry, size_int);
    cudaMalloc((void**)&d_distance, size_float*2);
	cudaMalloc((void**)&d_search, size_Mat_search);
	cudaMalloc((void**)&d_query, size_Mat_query);
    //cudaMalloc((void**)&d_path, size_vec);  --> des to segmentation fault (core dumped)

    // Copy inputs to device
    cudaMemcpy(numFeats, &feats, size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, &w, size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(lenQ, &rowsQ, size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(lenS, &rowsS, size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_version, &version, size_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_symmetry, &symmetry, size_int, cudaMemcpyHostToDevice);    
	cudaMemcpy( d_search, searchArr, size_Mat_search, cudaMemcpyHostToDevice);
	cudaMemcpy( d_query, queryArr, size_Mat_query, cudaMemcpyHostToDevice);

	cout << "asefwe f werf ...><" << query.data() << endl;

	// eigenMatrixData2CUDA( search->data, search->rows(), search->cols(), *d_search);
	// eigenMatrixData2CUDA( query->data, query->rows(), query->cols(), *d_query);

    steady_clock sc;
    auto start_timer = sc.now();
  	//DeTeWe_GPU<<1,1>>( query, search, w, version, symmetry, path);
  	Yi_bounDemLow_GPU<<<2,512>>>( d_query, d_search, lenQ, lenS, numFeats, d_distance);
  	auto end_timer = sc.now();

  	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess)
	{
   		fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
   		exit(-1);
	}
    
    cout << static_cast<duration<double>>(end_timer - start_timer).count() << '\n';

	// // Copy result back to host
	// cudaMemcpy(&path, d_path.data(), size_vec, cudaMemcpyDeviceToHost);
	cudaMemcpy( distDTW, d_distance, size_float, cudaMemcpyDeviceToHost);
	//cudaMemcpy( queryArr, &d_query, size_Mat_query, cudaMemcpyDeviceToHost);

	// // Cleanup
	cudaFree(d_search); cudaFree(d_query); cudaFree(d_symmetry); cudaFree(d_version); cudaFree(d_w); cudaFree(d_distance); //cudaFree(d_path); 
	
	//cout << "LowBound dist: " << *distDTW << endl;
	printf("LobaDist: %f \n\n", distDTW[0]);

	free(distDTW); //free(search); free(query); //free(version); free(symmetry); 

    return 0;

}

