

#include <vector>
#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/Splines>

//#include <cuda.h>
#include <cuda_runtime.h>

#include "cublas_v2.h"

using namespace std;
using namespace Eigen;


# define arr_sizeof(type) ((char *)(&type+1)-(char*)(&type)) 

float euclideanDist( VectorXf vec1, VectorXf vec2)
{

	int i ;
	float dist = 0.0;
	int numOfFeats = vec1.cols();

	for( i=0; i<numOfFeats; i++)
		dist += pow( (vec1(i)-vec2(i)), 2);
	
	dist = sqrt(dist);

	return dist;
}


// L_infinite distance aka Chebyshev distance 
float Linf_Dist( MatrixXf dat1, MatrixXf dat2)
{

	float linf_dist;

    if( dat1.rows()==4 )
    {
    	linf_dist = 0.0;
        linf_dist = max( { abs(dat1(0)-dat2(0)), abs(dat1(1)-dat2(1)), abs(dat1(2)-dat2(2)), abs(dat1(3)-dat2(3) ) } );
    }
    else if( dat1.rows()==2 )
    {
    	float maxXY_1stPnt = max( { abs(dat1(0,0)-dat2(0,0)), abs(dat1(0,1)-dat2(0,1))});
    	float maxXY_LastPnt = max( { abs(dat1(1,0)-dat2(1,0)), abs(dat1(1,1)-dat2(1,1))} );
        linf_dist = max(  maxXY_1stPnt, maxXY_LastPnt );

    }

    return linf_dist;
}


// __device__ Matrix2f *dat_query;
// __device__ Matrix2f *dat_candidate;

// L_infinite distance aka Chebyshev distance 
/*__device__ void Linf_Dist_GPU( Matrix2f dat1, Matrix2f dat2, float *linf_dist)
{
    printf("asas1");
	float linf_temp;
	if( dat1.rows()==4 )
    {
    	//linf_dist = 0.0;
        //linf_dist = max( { abs(dat1(0)-dat2(0)), abs(dat1(1)-dat2(1)), abs(dat1(2)-dat2(2)), abs(dat1(3)-dat2(3) ) } );
        *linf_dist = max( { abs(dat1(0)-dat2(0)), abs(dat1(1)-dat2(1)), abs(dat1(2)-dat2(2)), abs(dat1(3)-dat2(3) ) } );
        printf("asas2");
    }
    else if( dat1.rows()==2 )
    {
    	float maxXY_1stPnt = max( { abs(dat1(0,0)-dat2(0,0)), abs(dat1(0,1)-dat2(0,1))});
    	float maxXY_LastPnt = max( { abs(dat1(1,0)-dat2(1,0)), abs(dat1(1,1)-dat2(1,1))} );
        linf_temp = max(  {maxXY_1stPnt, maxXY_LastPnt} );
        linf_dist = &linf_temp;
        printf("Linf: %f ",linf_dist);
        printf("asas3");
    }

}*/

__device__ void Linf_Dist_GPU( float* dat1, float* dat2, size_t lenDats, float *linf_dist)
{
    //printf("asas1");
    float linf_temp;
    // if( lenDats==8 )
    // {
    //     //linf_dist = 0.0;
    //     //linf_dist = max( { abs(dat1(0)-dat2(0)), abs(dat1(1)-dat2(1)), abs(dat1(2)-dat2(2)), abs(dat1(3)-dat2(3) ) } );
    //     //*linf_dist = max( { abs(dat1(0)-dat2(0)), abs(dat1(1)-dat2(1)), abs(dat1(2)-dat2(2)), abs(dat1(3)-dat2(3) ) } );
    //     printf("asas2");
    // }
    // else if( lenDats==4 )
    // {
    //     //float maxXY_1stPnt = max( { abs(dat1(0,0)-dat2(0,0)), abs(dat1(0,1)-dat2(0,1))});
    //     //float maxXY_LastPnt = max( { abs(dat1(1,0)-dat2(1,0)), abs(dat1(1,1)-dat2(1,1))} );
    //int indx = 

    float maxXY_1stPnt = max( { abs(dat1[0]-dat2[0]), abs(dat1[1]-dat2[1])});
    float maxXY_LastPnt = max( { abs(dat1[2]-dat2[2]), abs(dat1[3]-dat2[3])} );
    linf_temp = max(  {maxXY_1stPnt, maxXY_LastPnt} );
    linf_dist[0] = linf_temp;
    //     //printf("Linf: %f ",linf_dist);
    //     //printf("asas3");
    // }

}



/*
paper: 
An index-basedapproach for similarity search supporting time warpingin large sequence databases
*/
// Kim lower bound features, that is first, last, lower and higher points 
float Kim_bounDemLow_4pnt( MatrixXf trajQ, MatrixXf trajC)
{    

    Vector4f dat_query = Vector4f::Zero();
    Vector4f dat_candidate = Vector4f::Zero();              
    float loba;
        
    VectorXf::Index higherQ, lowerQ, higherC, lowerC;
    float maxQ, minQ, maxC, minC;

	// extract higher, lower point (x,y) of query
	maxQ = trajQ.col(1).maxCoeff( &higherQ );
	minQ = trajQ.col(1).minCoeff( &lowerQ );
   
    // extract higher, lower point (x,y) of candidate
	maxC = trajC.col(1).maxCoeff( &higherC );
	minC = trajC.col(1).minCoeff( &lowerC );
    
    // left/right hand trajectory simplified as first, lower, higher and last point
    dat_query(0) = trajQ(0,1);
    dat_query(1) = trajQ(trajQ.rows()-1,1);
    dat_query(2) = trajQ(lowerQ,1);
    dat_query(3) = trajQ(higherQ,1); 
    
    dat_candidate(0) = trajC(0,1);
    dat_candidate(1) = trajC(trajC.rows()-1,1);
    dat_candidate(2) = trajC(lowerC,1);
    dat_candidate(3) = trajC(higherC,1); 

    loba = Linf_Dist( dat_query, dat_candidate);
    
	loba /= (trajQ.rows() + trajC.rows());

    return loba;

}


__device__ float linf_temp;
// __device__ float maxXY_1stPnt;
// __device__ float maxXY_LastPnt;
// Kim lower bound features, that is first and last point
__global__ void Kim_bounDemLow_2pnt_GPU( float *trajQ, float *trajC, int *lenTrajQ, int *lenTrajC, float *loba)
{

    //int indx = blockDim.x*blockIdx.x + threadIdx.x;

    float dat_query[4] = { trajQ[0], trajQ[0 + *lenTrajQ], trajQ[*lenTrajQ-1], trajQ[*lenTrajQ*2-1] };
    float dat_candidate[4] = { trajC[0], trajC[0 + *lenTrajC], trajC[*lenTrajC-1], trajC[*lenTrajC*2-1] };

    size_t lenDat = sizeof(dat_query)/sizeof(dat_query[0]);

    Linf_Dist_GPU( dat_query, dat_candidate, lenDat, loba);
    //printf("Loba: %f",*loba);
    
    int normalByRows = *lenTrajQ + *lenTrajC;
    loba[0] /= normalByRows;
    //loba[1] /= normalByRows;
    //printf("Loba: %f \n",*loba);
    
}
    

__global__ void Yi_bounDemLow_GPU( float *trajQ, float *trajC, int *lenTrajQ, int *lenTrajC, int *numFeats, float *loba)
{

    float maxQ_[2], minQ_[2];
    int i, j;
    float sum_squared_ = 0.0;

    float maxT, minT;
    for( i=0; i<*numFeats; i++)
    {   
        maxT = 0.0, minT = 100000.0;
        for( j=*lenTrajQ*i; j<*lenTrajQ*(i+1); j++)
        {
            maxT = (trajQ[j]<maxT) ? maxT:trajQ[j];
            minT = (trajQ[j]>minT) ? minT:trajQ[j];    
        }
        maxQ_[i] = maxT;
        minQ_[i] = minT;
    }
    
    for( i=0; i<*numFeats; i++)
    {   
        for( j=*lenTrajC*i; j<*lenTrajC*(i+1); j++){

            if( trajC[j] > maxQ_[i] )
                sum_squared_ += abs(trajC[j] - maxQ_[i]);
                
            if( trajC[j] < minQ_[i] )
                sum_squared_ += abs(minQ_[i] - trajC[j]);
        }                               
    }
    
    loba[0] = sum_squared_ / (*lenTrajQ + *lenTrajC);
    
}


/*
paper:
Efficient retrieval of similar time sequences under time warping
*/
// Yi lower bound is the sum of the length between candidate
// points that exceed query's higher and lower curve points
float Yi_bounDemLow( MatrixXf trajQ, MatrixXf trajC)
{
    


    float loba = 0.0;
    Matrix2f max_minQ(trajQ.cols(),2);        
    int i, j, numRows = trajC.rows(), numCols = trajC.cols();
        
    // extract higher, lower point (x,y) of left/right hand trajectory
	for( i=0; i<numCols; i++)
	{
		max_minQ(i,0) = trajQ.col(i).maxCoeff();
		max_minQ(i,1) = trajQ.col(i).minCoeff();
	}

	float sum_squared_ = 0.0;
    for( i=0; i<numRows; i++)
    {   
    	for( j=0; j<numCols; j++){

	        if( trajC(i,j) > max_minQ(j,0) )
	            sum_squared_ += abs(trajC(i,j) - max_minQ(j,0));
	            
	        if( trajC(i,j) < max_minQ(j,1) )
	            sum_squared_ += abs(max_minQ(j,1) - trajC(i,j));
        }                               
    }
    
    loba = sum_squared_ / (trajQ.rows() + trajC.rows());
	
    return loba;

}


void findUpperLowerEnvelope( MatrixXf traj, int w, MatrixXf &uppEnv, MatrixXf &lowEnv)
{
    int lenT = traj.rows();
    int numFeats = traj.cols();
    int i, j, range;
    
    for( i=w; i<lenT-w; i++)
    {
    	range = i-w;
        for( j=0; j<numFeats; j++)
        {
            uppEnv(i,j) = traj.block(range,j,2*w,1).maxCoeff();
            lowEnv(i,j) = traj.block(range,j,2*w,1).minCoeff();
    	}
    }
    
    //copy w-th value to the first w elements & last-w value to last-w elements
    for( i=0; i<w; i++)
    {
    	for( j=0; j<numFeats; j++)
    	{
	        uppEnv(i,j) = uppEnv(w,j);
	    	lowEnv(i,j) = lowEnv(w,j);

	    	uppEnv(lenT-w+i,j) = uppEnv(lenT-w-1,j);    	
	    	lowEnv(lenT-w+i,j) = lowEnv(lenT-w-1,j);
	    }
    }

}


/*
paper:
Exact  indexing  of  dynamic  time  warping
*/
float Keogh_bounDemLow_SakoeChiba( MatrixXf trajC, MatrixXf trajQ, int w, MatrixXf &upperQ, MatrixXf &lowerQ)
{
    int lenQ = trajQ.rows(), lenQ_cols = trajQ.cols();
    int lenC = trajC.rows(), lenC_cols = trajC.cols();
    int i, j;
    float sum_cand_ULQ = 0.0, loba = 0.0;


    Vector2f maxCoe, minCoe;
    maxCoe(0) = -1000.0;
    maxCoe(1) = -1000.0;
    minCoe(0) = 1000.0;
    minCoe(1) = 1000.0;
   
    MatrixXf uppEnv = MatrixXf::Zero( trajQ.rows(), trajQ.cols()); 
    MatrixXf lowEnv = MatrixXf::Zero( trajQ.rows(), trajQ.cols()); 
    findUpperLowerEnvelope( trajQ, w, uppEnv, lowEnv);
    

    sum_cand_ULQ = 0;
    for( i=0; i<lenC; i++)
    {
    	for( j=0; j<lenC_cols; j++)
    	{
            if( trajC(i,j) > upperQ(i,j) ){
                sum_cand_ULQ += pow(trajC(i,j) - upperQ(i,j),2);
            }
                
            else if( trajC(i,j) < lowerQ(i,j) ){
                sum_cand_ULQ += pow(trajC(i,j) - lowerQ(i,j),2);
            }
        }
    }

    loba = sqrt(sum_cand_ULQ) / (lenQ + lenC);

    return loba;
}


/*
paper:
Faster retrieval with a two-pass dynamic-time-warping lower bound
*/
float Improved_bounDemLow( MatrixXf trajC, MatrixXf trajQ, int w, MatrixXf &uppQ, MatrixXf &lowQ)
{

	int i, j;
	float loba = 0.0;

    MatrixXf uppEnvQ = MatrixXf::Zero( trajQ.rows(), trajQ.cols()); 
    MatrixXf lowEnvQ = MatrixXf::Zero( trajQ.rows(), trajQ.cols()); 
    findUpperLowerEnvelope( trajQ, w, uppEnvQ, lowEnvQ);

    int lenQ = trajQ.rows();
    int lenC = trajC.rows(), lenC_cols = trajC.cols();
    
    MatrixXf trajC_n = trajC; 
    for( i=0; i<lenC; i++)
    {
        for(j=0; j<lenC_cols; j++)
        {
            if(trajC(i,j) >= uppEnvQ(i,j))
                trajC_n(i,j) = uppEnvQ(i,j);
            else if(trajC(i,j) <= lowEnvQ(i,j))
                trajC_n(i,j) = lowEnvQ(i,j);
    	}
    }

    // cout << "UppQ: " << uppQ << endl << "LowQ: " << lowQ << endl;
    // cout << "TrajC: " << uppQ << endl << "TrajC_n: " << trajC_n << endl;
    MatrixXf upperC = MatrixXf::Zero( trajQ.rows(), trajQ.cols()); 
	MatrixXf lowerC = MatrixXf::Zero( trajQ.rows(), trajQ.cols()); 
    loba = Keogh_bounDemLow_SakoeChiba( trajQ, trajC, w, upperC, lowerC);

    return loba ;
}


/*
paper: 
Elastic bands across the path:A new framework and method to lower bound DTW

	uppQ: upper envelope of query series (trajQ)
	lowQ: lower envelope of query series (trajQ)
	w: warping window
	v: tightness parameter specifying the number of left and right bands
	d: current distance to Nearest Neighbour 
*/
float Enhanced_bounDemLow( MatrixXf trajC, MatrixXf trajQ, int w, int v, float d)
{
    //dist = lambda x, y: np.sqrt( np.dot(x, x) - 2*np.dot(x, y) + np.dot(y, y) ) 

    MatrixXf uppEnvQ = MatrixXf::Zero( trajQ.rows(), trajQ.cols()); 
    MatrixXf lowEnvQ = MatrixXf::Zero( trajQ.rows(), trajQ.cols()); 
    findUpperLowerEnvelope( trajQ, w, uppEnvQ, lowEnvQ);

    int i, j;
    float minL, minR;
    int lenQ = trajQ.rows() - 1;
    int lenC = trajC.rows() - 1;
    int numFeats = trajC.cols();

    int rightEnd, rightStart;
    int L = trajC.rows() - 2;
    int numOfBands = min( (int)(L/2), v);

    float res = euclideanDist( trajC.row(0), trajQ.row(0)) + euclideanDist( trajC.row(lenC-2), trajQ.row(lenQ-2));
    for( i=1; i<numOfBands; i++) 
    {	
    	rightEnd = L-i;
    	minL = euclideanDist( trajC.row(i), trajQ.row(i));
    	minR = euclideanDist( trajC.row(L-i), trajQ.row(L-i));
    	for( j=max(1,i-w); j<i-1; j++)
    	{
    		rightStart = L-j;
    		minL = min( minL, euclideanDist( trajC.row(i), trajQ.row(j)));
    		minL = min( minL, euclideanDist( trajC.row(j), trajQ.row(i)));

    		minR = min( minR, euclideanDist( trajC.row(rightEnd), trajQ.row(rightStart)));
    		minR = min( minR, euclideanDist( trajC.row(rightStart), trajQ.row(rightEnd)));
    	}
    	res += minL + minR;
    }

    if((res/(lenQ+lenC))>=d)
    	return 100000.0;

    float temp_res = 0.0;
    for( i=(numOfBands); i<=(L-numOfBands); i++)
    {
    	for( j=0; j<numFeats; j++)
    	{
			if (trajC(i,j) > uppEnvQ(i,j))
	            temp_res += pow((trajC(i,j) - uppEnvQ(i,j)),2);
	            
	        else if(trajC(i,j) < lowEnvQ(i,j))
	            temp_res += pow((trajC(i,j) - lowEnvQ(i,j)),2);
	    }
    }
    res = (res + sqrt(temp_res)) / (lenQ + lenC);

    return res;

}



float Enhanced1_bounDemLow( MatrixXf trajC, MatrixXf trajQ, int w, int v, float d)
{

    int lenQ = trajQ.rows();
    int lenC = trajC.rows();
    int numFeats = trajC.cols();
    int i, j;

    MatrixXf uppEnvQ = MatrixXf::Zero( trajQ.rows(), trajQ.cols()); 
    MatrixXf lowEnvQ = MatrixXf::Zero( trajQ.rows(), trajQ.cols()); 
    findUpperLowerEnvelope( trajQ, w, uppEnvQ, lowEnvQ);

    int L = trajC.rows() - 2;
    float res = euclideanDist( trajC.row(0), trajQ.row(0)) + euclideanDist( trajC.row(lenC-2), trajQ.row(lenQ-2));

    if((res/(lenQ+lenC))>=d)
    	return 100000.0;
    
    float temp_res = 0;
    for( i=1; i<L; i++)
    {
    	for( j=1; j<numFeats; j++) 
    	{
    		if(trajC(i,j) > uppEnvQ(i,j))
	            temp_res += pow(trajC(i,j) - uppEnvQ(i,j),2);
	            
    		else if(trajC(i,j) < lowEnvQ(i,j))
	            temp_res += pow(trajC(i,j) - lowEnvQ(i,j),2);
    	}
    }

    res = (res + sqrt(temp_res)) / (lenC + lenQ);
    
    return res;

}

