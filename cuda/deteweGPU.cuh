

#include <vector>
#include <iostream>
#include <limits>

#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

#define lenTemp 3

float evaluateAlignment( float Dp, vector<vector<int>> path)
{
	// constant that helps bigger commonalities 
	// be more important to smaller ones
	float Fp, c = 0.1;

	// find diagonal elements
	int matches = 0;
	int i;
	for( i=0; i<path.size(); i++)
	{
		if(path[i][0] == path[i][1])
			matches += 1;
	} 

	// Fitness score - weighted average evaluation of DTW distance
	Fp = (Dp + c)/(matches + 1);

	return Fp;
}


// first order differential:
// provides shape information
__device__ float* firstDerivative( float toderiv[lenTemp])
{   
	float v_1st_deriv = ( (toderiv[1] - toderiv[0]) + ((toderiv[2] - toderiv[0])/2) ) / 2;
    
    float *v_ret = &v_1st_deriv;
    printf("VRet:: %f", *v_ret);

    return v_ret;
}

// second order differential:
// provides min value, max value, infection point information   
__device__ float* secondDerivative( float toderiv[lenTemp])
{
	float v_2nd_deriv = toderiv[2] + toderiv[0] - toderiv[0]*2;
    float *v_ret = &v_2nd_deriv;
    printf("VRet:: %f", *v_ret);
    
    return v_ret;
}	

//__device__ void euclideanDist( float *s, float *t, int dim, float *eucl_dist)
__device__ float euclideanDist( float *s, float *t, int dim)
{
	float eucl_dist = 0.0;
	
	if(dim == 2)
		eucl_dist = sqrt( pow(s[0] - t[0],2) + pow(s[1] - t[1],2));
	else if(dim == 3)
		eucl_dist = sqrt( pow(s[0] - t[0],2) + pow(s[1] - t[1],2)  + pow(s[2] - t[2],2));
	else if(dim == 1)
		eucl_dist = sqrt( pow(*s - *t, 2) );

	return eucl_dist;
}


/*

  Input Arguments:
 	- s: query
    - t: template
    - version: 0 -> classic distance matrix computation
			   1 -> distance matrix computation based on 1st,2nd derivatives
      (paper: Human Action Recognition based on DTW and Movement Trajectory)
    - symmetry: 0 -> symmetric, 1 -> quasi-symmetric, 2-> assymetric
    - w: warping window

  Output Arguments:
    - DTW[-1,-1]: DTW normalised distance
    - cost_matrix: distance matrix
    - DTW: cumulative distance matrix
    - path: DTW distance path

*/
__global__ void DeTeWe_GPU( float *que, float *sear, int lenQ, int lenS, int numFeats, int w, int version, int symmetry, float *dist, float *cost_matrix, float *dtw_matrix)
{
    
    int i, j, z;//, row_t, col_t;

    int row = blockIdx.y*blockDim.y + threadIdx.y; 
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    // warping window to constrain DTW path computation near the diagonal
    int warp = max( w, abs(lenQ - lenS));

    int count = 0;

    float *s_temp = new float[numFeats];
    float *t_temp = new float[numFeats];
 
    if(version==0)
    {          
        // for ( i=0; i<lenQ; i++) 
        // {
        //     for ( j=0; j<lenS; j++)
        //     {
        //         float s_temp[2] = { que[i], que[i + lenQ]}; //s[row_t + *lenS*z];
        //         float t_temp[2] = { sear[j], sear[j + lenS]}; //t[row_t + *lenT*z];
        //         cost_matrix[lenQ*(row+i)+(col+j)] = euclideanDist( s_temp, t_temp, numFeats); 

        //         // initialise dtw cumulative distance values with +inf
        //         dtw_matrix[lenQ*(row+i) + (col+j)] = numeric_limits<float>::max();

        //         count++;
        //     }
        //     //printf("\n");
        // }   
        if(row<lenQ)
        {
            for ( j=0; j<lenS; j++)
            {
                float s_temp[2] = { que[row], que[row + lenQ]}; //s[row_t + *lenS*z];
                float t_temp[2] = { sear[j], sear[j + lenS]}; //t[row_t + *lenT*z];
                cost_matrix[lenQ*(row)+(j)] = euclideanDist( s_temp, t_temp, numFeats); 
                //printf("CoMa: %f \t", cost_matrix[lenQ*(row)+(j)]);
                // initialise dtw cumulative distance values with +inf
                dtw_matrix[lenQ*(row) + (j)] = numeric_limits<float>::max();

                count++;
            }
        }
    }
    // for ( i=0; i<lenQ; i++) 
    // {
        
    //         printf("CoMa: %f \t", cost_matrix[lenQ*(i)+(j)]);
    //         //printf("DiMa: %f \t", cost_matrix[lenQ*(row)+(col)]);

    //         count++;
    //     }
    //     printf("\n");
    // }

            //printf("\n");
        


    //}    
/*    else if(version==1)
    {

        float s_temp0[2] = { que[0], que[0 + lenQ]}; //s[row_t + lenS*z];
        float t_temp0[2] = { sear[0], sear[0 + lenS]}; //t[row_t + lenT*z];
        cost_matrix[lenS*0 + 0] = euclideanDist( s_temp0, t_temp0, numFeats);

        float s_tempL[2] = { que[lenQ - 1], que[lenQ*2 - 1]}; //s[row_t + lenS*z];
        float t_tempL[2] = { sear[lenS - 1], sear[lenS*2 - 1]}; //t[row_t + lenT*z]; 
        cost_matrix[lenQ*(lenQ-1) + (lenS-1)] = euclideanDist( s_tempL, t_tempL, numFeats); 
        printf("ena,, ");
         // dtw derivative weights
        float w3 = 2.4, w2 = 1.2, w1 = 0.6;
        int count = 0;
        for( i=1; i<(lenQ - 1); i++)
        {
            printf("dyo,, ");
            if (que[i-1]==NULL || que[i]==NULL || que[i+1]==NULL)
                continue;

            count++;
            printf("Cntd: %d", count);
            float que_temp[2] = { que[i], que[i + lenQ] };

            float s_temp[3] = { que[i-1], que[i], que[i+1]}; //s[row_t + lenS*z];
            printf("ena2,, ");
            float *s_deriv1 = firstDerivative( s_temp );
            printf("ena3,, ");
            float *s_deriv2 = secondDerivative( s_temp );
            int count1 = 0;
            for( j=1; j<(lenS - 1); j++)
            {
                printf("tria,, ");
                float t_temp[3] = { sear[j-1], sear[j], sear[j+1]}; //s[row_t + lenS*z];
                float sear_temp[2] = { sear[j], sear[lenS + j] };

                count1++;
                printf("Cntd1!!!: %d", count1);
                float *t_deriv1 = firstDerivative( t_temp );
                float *t_deriv2 = secondDerivative( t_temp );
                cost_matrix[lenQ*i+j] = w1*euclideanDist( que_temp, sear_temp, numFeats) + w2*euclideanDist( s_deriv1, t_deriv1, 1) + w3*euclideanDist( s_deriv2, t_deriv2, 1); 
                //cost_matrix[*lenT*row+col] = w2*euclideanDist( s_deriv1, t_deriv1, 1) + w3*euclideanDist( s_deriv2, t_deriv2, 2);       
                printf("CMat: %f \t", cost_matrix[lenQ*i+j]);
            }
            printf("\n");
        }
    }*/


    int row1 = blockIdx.y*blockDim.y + threadIdx.y; 
    int col1 = blockIdx.x*blockDim.x + threadIdx.x;

    // if symmetric DTW
    if( symmetry == 0 )
    {
        dtw_matrix[lenQ*0 + 0] = 2*cost_matrix[lenQ*0 + 0];
        //int countDTW1 = 0, countDTW2 = 0, countDTW3 = 0;
        if (row1>=1 && row1<min(lenQ,warp+1))
            dtw_matrix[lenQ*row1 + 0] = dtw_matrix[lenQ*(row1-1) + 0] + cost_matrix[lenQ*row1 + 0];

        if (col1>=1 && col1<min(lenS,warp+1))
            dtw_matrix[lenQ*0 + col1] = dtw_matrix[lenQ*0 + (col1-1)] + cost_matrix[lenQ*0 + col1];

        // for( i=(row+1); i<min(lenQ,warp+1); i++) 
        // {
        //     dtw_matrix[lenQ*i+0] = dtw_matrix[lenQ*(i-1) + 0] + cost_matrix[lenQ*i + 0];
        // }
        // for( i=(col+1); i<min(lenS,warp+1); i++)
        // {
        //     dtw_matrix[lenQ*0 + i] = dtw_matrix[lenQ*0 + (i-1)] + cost_matrix[lenQ*0 + i];
        //     //countDTW2++;
        // }
        
        for( i=1; i<lenQ; i++)
        {
            for( j=max( 1, i-warp); j<min( lenS, i+warp+1); j++)
            {
                dtw_matrix[lenQ*(row+i)+(col+j)] = min( { dtw_matrix[ lenQ*(row+i-1) + (col+j) ] + cost_matrix[ lenQ*(row+i) + (col+j)], dtw_matrix[ lenQ*(row+i) + (col+j-1)] + cost_matrix[ lenQ*(row+i) + (col+j)], dtw_matrix[ lenQ*(row+i-1) + (col+j-1)] + 2*cost_matrix[ lenQ*(row+i) + (col+j)] } );
                //countDTW3++;
                               
                //printf("dtwMat: %f \t", dtw_matrix[lenQ*(row+i)+(col+j)]);

            }
            //printf("\n");

        }
        // time-normalised distance
        dist[0] = dtw_matrix[lenQ*(lenQ-1) + (lenS-1)] / (lenQ+lenS);

    }
    // if quazi-symmetric DTW
    else if( symmetry == 1 )
    {
        dtw_matrix[lenQ*0 + 0] = cost_matrix[lenQ*0 + 0];
        if (row1>=1 && row1<min(lenQ,warp+1))
            dtw_matrix[lenQ*row1 + 0] = dtw_matrix[lenQ*(row1-1) + 0] + cost_matrix[lenQ*row1 + 0];

        if (col1>=1 && col1<min(lenS,warp+1))
            dtw_matrix[lenQ*0 + col1] = dtw_matrix[lenQ*0 + (col1-1)] + cost_matrix[lenQ*0 + col1];

        // for( i=1; i<min(lenQ,warp + 1); i++)
        //     dtw_matrix[lenQ*i + 0] = dtw_matrix[lenQ*(i-1) + 0] + cost_matrix[lenQ*i + 0];

        // for( i=1; i<min(lenS,warp + 1); i++)
        //     dtw_matrix[ lenQ*0 + i] = dtw_matrix[ lenQ*0 + (i-1)] + cost_matrix[lenQ*0 +i];
        
        for( i=1; i<lenQ; i++)
        {    
            for( j=max( 1, i-warp); j<min( lenS, i+warp+1); j++)
            {
                dtw_matrix[ lenQ*i + j] = cost_matrix[ lenQ*i + j] + (float)min( { dtw_matrix[ lenQ*(i-1) + j],  dtw_matrix[ lenQ*i + (j-1)], dtw_matrix[ lenQ*(i-1) + (j-1)] } );
                //printf("dtwMat: %f", dtw_matrix[ lenQ*i + j]);
            }           
            //printf("\n");
        }
        // time-normalised distance
        //dtw_matrix[ lenQ*(lenQ-1) + (lenS-1)] /= (lenQ + lenS); 
        dist[0] = dtw_matrix[lenQ*(lenQ-1) + (lenS-1)] / (lenQ+lenS);
        //printf("dist: %f", dist[0]);

    }
    // if asymmetric DTW
    else if( symmetry == 2 )
    {

        dtw_matrix[lenQ*0 + 0] = cost_matrix[lenQ*0 + 0];

        if (row1>=1 && row1<min(lenQ,warp+1))
            dtw_matrix[lenQ*row1 + 0] = dtw_matrix[lenQ*(row1-1) + 0] + cost_matrix[lenQ*row1 + 0];

        if (col1>=1 && col1<min(lenS,warp+1))
            dtw_matrix[lenQ*0 + col1] = dtw_matrix[lenQ*0 + (col1-1)] + cost_matrix[lenQ*0 + col1];

        // for( i=1; i<min(lenQ,warp + 1); i++)
        //     dtw_matrix[lenQ*i + 0] = dtw_matrix[lenQ*(i-1) + 0] + cost_matrix[lenQ*i + 0];
        
        // for( i=1; i<min(lenS,warp + 1); i++)
        //     dtw_matrix[lenQ*0 + i] = dtw_matrix[lenQ*0 + (i-1)] + cost_matrix[lenQ*0 + i];

        for( i=1; i<lenQ; i++)
            for( j=max( 1, i-warp); j<min( lenS, i+warp+1); j++)  //(1,m):
                dtw_matrix[lenQ*i + j] = min( { dtw_matrix[lenQ*(i-1) + j] + cost_matrix[lenQ*i + j], dtw_matrix[lenQ*i + (j-1)], dtw_matrix[lenQ*(i-1) + (j-1)] + cost_matrix[lenQ*i + j] } );
            
        
        // time-normalised distance
        //dtw_matrix[lenQ*(lenQ-1) + (lenS-1)] /= (lenQ);
        dist[0] = dtw_matrix[lenQ*(lenQ-1) + (lenS-1)] / (lenQ);
        
    }

    // auto end_timer = sc.now();
    // cout << static_cast<duration<double>>(end_timer - start_timer).count() << '\n';
    
    // backtrace to find path
    // i = DTW_matrix.rows() - 2;
    // j = DTW_matrix.cols() - 2;
 
    //ArrayXXd path(n,m);
    //vector<vector<int>> path;
    
    //path.resize(n);
    //list<list<int>> path(n,m);
    /*int min_ind, cnt = 0;
    while((i >= 0) && (j >= 0))
    {
        vector<int> path_matches;
        if(i==0 && j>0)
        {
            path_matches.push_back(i);
            path_matches.push_back(j);
            j = j - 1;
        }
        else if( j==0 && i>0)
        {
            path_matches.push_back(i);
            path_matches.push_back(j);
            i = i - 1;
        }
        else
        {

            float neighborhood3[] = { DTW_matrix(i, j), DTW_matrix(i, j + 1), DTW_matrix(i + 1, j)};
            const int N = sizeof(neighborhood3) / sizeof(float);
            min_ind = distance( neighborhood3, min_element(neighborhood3, neighborhood3+N));
            
            if(min_ind == 0)
            {
                path_matches.push_back(i);
                path_matches.push_back(j);
                i -= 1;
                j -= 1;
            }
            else if(min_ind == 1)
            {
                path_matches.push_back(i);
                path_matches.push_back(j+1);
                i -= 1;
            }
            else
            {  
                path_matches.push_back(i+1);
                path_matches.push_back(j);
                j -= 1;
            }
        }
        path.push_back(path_matches);
    }
    reverse( path.begin(), path.end());
    for( unsigned int x=0; x<path.size(); x++)
    {
        for( unsigned int y=0; y<path[x].size(); y++)
        {   
            cout << path[x][y] << " ";
        }
        cout << endl;
    }*/
   
    //return DTW_matrix(n-1,m-1);*/
}
