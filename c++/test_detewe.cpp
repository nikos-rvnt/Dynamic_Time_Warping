
//g++ test_detewe.cpp -o test_detewe -lstdc++fs -std=c++17
#include <fstream>
#include <nlohmann/json.hpp>

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <experimental/filesystem>

#include <eigen3/Eigen/Dense>

//#include "detewe.cpp"
#include "detewe.h"
#include "lower_bounds.h"
#include "splineInterpolation.h"

namespace fs = std::experimental::filesystem;

using json = nlohmann::json;
using namespace std;
using namespace Eigen;
using namespace chrono;



int main()
{

	string pathAnno = "/media/nikos/Data/NAS_healthPose/Athens 2019-07-23/HealthSign_ximea/24_07/";
	vector<string> pathGlossAnno;
	for (const auto & entry : fs::directory_iterator(pathAnno))
        pathGlossAnno.push_back(entry.path());

    //cout << pathGlossAnno[0] << endl;
    ifstream glossAnno1(pathGlossAnno[0]);
    //cout << pathGlossAnno[0] << endl;
	json j1;
	j1 = json::parse(glossAnno1);

	// typedef Matrix<double, Dynamic, 1> VectorXi;
	// VectorXi vec_temp;

    int i, j, w, i_temp, version = 0, symmetry = 1;
	// vector<float*> s_j1, t_j1;
	
	// auto temp = j1["joints_2d_right"][0];
	// cout << temp << '\n' << temp.size();
	// //auto temp = j1["joints_2d_right"][0].get_to(s_temp);

	// //cout << typeid(temp).name() << endl;
	// //s_j1.push_back(temp);

	// vector<vector<double>> vecf;
	// for( i=48; i<50; i++)
	// {
	//  	auto temp = j1["joints_2d_right"][i];
	//  	cout << temp << endl << endl;
	//  	vector<double> vecf1;
	//  	for ( j=8; j<10; j++){
	//  		vecf1.push_back((double)temp[j]);
	//  	}
	//  // 	cout << temp << endl;
	//  	vecf.push_back(vecf1);
	// 	// s_j1.push_back(j1["joints_2d_right"][i]);
	// 	// cout << endl << j1["joints_2d_right"][i];
	// 	//s_j1.push_back((ArrayXf)j1["joints_2d_right"][i]);
	// }
	// cout << vecf[0][0] << endl;
	// for( i=48; i<82; i++)
	// 	t_j1.push_back((ArrayXf)j1["joints_2d_right"][i]);

	// cout << s_j1.size();

    float lb_dist = 0.0;
    double x4, y4;
    MatrixXf search(128,2), query(128,2);

    i_temp = 0;
    for( i=48; i<176; i++){

    	// extract 4th point (x,y) 
		x4 = (double)j1["joints_2d_right"][i][6];
		y4 = (double)j1["joints_2d_right"][i][7];    	

		query(i_temp,0) = x4;
		query(i_temp,1) = y4;
		i_temp++;
    }

    i_temp = 0;
    for( i=88; i<216; i++){

		x4 = (double)j1["joints_2d_right"][i][6];
		y4 = (double)j1["joints_2d_right"][i][7];    	

		search(i_temp,0) = x4;
		search(i_temp,1) = y4;
    	i_temp++;
    }


    w = abs( search.rows() - query.rows()) + 25;
    w = (int)(0.2*query.rows());
    //w = 10;
    int lobidi = 1;

 //    steady_clock sc;
 //    auto start_timer = sc.now();

 //    if(lobidi==0)
	// {
	// 	lb_dist = Kim_bounDemLow_2pnt( query, search);
	// }
 //    else if(lobidi==1)
	// {
	// 	steady_clock sc1;
	// 	auto start_timer = sc1.now();
	//     lb_dist = Yi_bounDemLow( search, query);
	//     auto end_timer = sc1.now();
	// 	cout << static_cast<duration<double>>(end_timer - start_timer).count() << '\n';
	// 	cout << endl << "LowedDEm" << endl; 
	// }
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
    //vector<std::chrono::duration<double>> timer(10);
    auto sum = std::chrono::duration<double>(0.0);
    for ( i=0; i<10; i++)
    {

	    steady_clock sc;
	    auto start_timer = sc.now();
	  	float distanceDTW = DeTeWe( query, search, w, version, symmetry, path);
	  	auto end_timer = sc.now();
	  	auto time_diff = end_timer - start_timer; 
	  	//timer.push_back(static_cast<duration<double>>(time_diff));
	  	sum += static_cast<duration<double>>(time_diff);
	    cout << static_cast<duration<double>>(end_timer - start_timer).count() << '\n';
	}
    cout << "Mean time: " << sum / static_cast<duration<double>>(10) ;

  	// cout << "DTW dist: " << distanceDTW << endl;
  	// for( unsigned int x=0; x<path.size(); x++)
   //  {
   //  	for( unsigned int y=0; y<path[x].size(); y++)
   //  	{	
   //  		cout << path[x][y] << " ";
   //  	}
   //  	cout << endl;
   //  }


    return 0;

}

