

#include <vector>
#include <iostream>

#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;


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
// float* firstDerivative( vector<float> *v)
// {
// 	float v_1st_deriv = ( (v[1] - v[0]) + ((v[2] - v[0])/2) ) / 2;

// 	return v_1st_deriv
// }

// // second order differential:
// // provides min value, max value, infection point information   
// float secondDerivative( vector<float> *v)
// {
// 	float v_sec_deriv = v[2] + v[0] - 2*v[0];

// 	return v_sec_deriv
// }	

float euclideanDist( auto s, auto t, int dim)
{
	auto eucl_dist = 0.0;
	//eucl_dist = s.transpose()*s -2*s.transpose()*t + t.transpose()*t;

	if(dim == 2)
		eucl_dist = sqrt( pow(s[0] - t[0],2) + pow(s[1] - t[1],2));
	else if(dim == 3)
		eucl_dist = sqrt( pow(s[0] - t[0],2) + pow(s[1] - t[1],2)  + pow(s[2] - t[2],2));
	else if(dim == 1)
		eucl_dist = sqrt( pow(s[0] - t[0],2) );

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
float DeTeWe( auto s, auto t, int w, int version, int symmetry, vector<vector<int>> &path)
{
    
    int i, j;
    int n = s.rows();
    int m = t.rows();

    // warping window to constrain DTW path computation near the diagonal
    int warp = max( w, abs(n-m));
    
    // cost_matrix -> distance matrix
    MatrixXf cost_matrix = MatrixXf::Zero( n, m); //cost_matrix = np.zeros((n,m))
    if( version == 0 )
    {
    	for( i=0; i<n; i++)
    		for( j=0; j<m; j++)
				cost_matrix(i,j) = euclideanDist( s.row(i), t.row(j), 2);
				//cost_matrix(i,j) = (s(i).rowwise() - t(j).row(s)).matrix().rowwise().norm();
    }
 //    }
 //    else
 //    {
 //        cost_matrix[0,0] = dist( s[0], t[0])
 //        cost_matrix[n-1,m-1] = dist( s[n-1], t[m-1])
 //   		// dtw derivative weights
 //        float w3 = 2.4, w2 = 1.2, w1 = 0.6;
 //        for( i=1; i<n-1; i++)
 //        {
 //            s_deriv1 = firstDerivative(s[i-1:i+2]);
 //            s_deriv2 = secondDerivative(s[i-1:i+2]);
 //            for( j=1; j<m-1; j++)
 //         	{       
 //                t_deriv1 = firstDerivative(t[j-1:j+2]);
 //                t_deriv2 = secondDerivative(t[j-1:j+2]);
 //                cost_matrix[i,j] = w1*dist(s[i],t[j]) + w2*dist(s_deriv1,t_deriv1) + w3*dist(s_deriv2,t_deriv2);
	// 		}
	// 	}
	// }

	// cumulative distance matrix
    // DTW = np.zeros((n,m))
    //cout << cost_matrix << endl;
    MatrixXf DTW_matrix = MatrixXf::Zero( n, m);

    // if symmetric DTW
    if( symmetry == 0 )
    {
        DTW_matrix(0,0) = 2*cost_matrix(0,0);
        for( i=1; i<min(n,warp+1); i++) 
        {
            DTW_matrix(i,0) = DTW_matrix(i-1,0) + cost_matrix(i,0);
            //DTW[i,0] = np.sum( cost_matrix[:i+1,0] )
    	}
        for( i=1; i<min(m,warp+1); i++)
        {
            DTW_matrix(0,i) = DTW_matrix(0,i-1) + cost_matrix(0,i);
            //DTW[0,i] = np.sum( cost_matrix[0,:i+1] )
        }
        
        for( i=1; i<n; i++)
        {
            for( j=max( 1, i-warp); j<min( m, i+warp+1); j++)
            {

                DTW_matrix(i,j) = min( { DTW_matrix(i-1,j) + cost_matrix(i,j), DTW_matrix(i,j-1) + cost_matrix(i,j), DTW_matrix(i-1,j-1) + 2*cost_matrix(i,j) } );
            }
        }
        // time-normalised distance
        DTW_matrix(n-1,m-1) = DTW_matrix(n-1,m-1) / (n+m);
    }
    // if quazi-symmetric DTW
    else if( symmetry == 1 )
    {
        DTW_matrix(0,0) = cost_matrix(0,0);
        for( i=1; i<min(n,warp + 1); i++)
        {
            DTW_matrix(i,0) = DTW_matrix(i-1,0) + cost_matrix(i,0);
            //DTW[i,0] = np.sum( cost_matrix[:i+1,0] )
        }
        for( i=1; i<min(m,warp + 1); i++)
        {
            DTW_matrix(0,i) = DTW_matrix(0,i-1) + cost_matrix(0,i);
            //DTW[0,i] = np.sum( cost_matrix[0,:i+1] )
		}

        for( i=1; i<n; i++)
        {
            for( j=max( 1, i-warp); j<min( m, i+warp+1); j++)  
            //for j in range(1,m):
            {
            	
                DTW_matrix(i,j) = cost_matrix(i,j) + (float)min( { DTW_matrix(i-1,j),  DTW_matrix(i,j-1), DTW_matrix(i-1,j-1) } );
            }
        }
        // time-normalised distance
        DTW_matrix(n-1,m-1) = DTW_matrix(n-1,m-1) / (n+m); 
        //cout << DTW_matrix << endl;
        //cout << DTW_matrix(n-1,m-1) << endl;
    }
    // if asymmetric DTW
    else if( symmetry == 2 )
    {

        DTW_matrix(0,0) = cost_matrix(0,0);
        for( i=1; i<min(n,warp + 1); i++)
        {
            DTW_matrix(i,0) = DTW_matrix(i-1,0) + cost_matrix(i,0);
            //DTW[i,0] = np.sum( cost_matrix[:i+1,0] )
        }
        for( i=1; i<min(m,warp + 1); i++)
        {
            DTW_matrix(0,i) = DTW_matrix(0,i-1) + cost_matrix(0,i);
            //DTW[0,i] = np.sum( cost_matrix[0,:i+1] )
		}

        for( i=1; i<n; i++)
        {
            for( j=max( 1, i-warp); j<min( m, i+warp+1); j++)  //(1,m):
            {
                DTW_matrix(i,j) = min( { DTW_matrix(i-1,j) + cost_matrix(i,j),  DTW_matrix(i,j-1), DTW_matrix(i-1,j-1) + cost_matrix(i,j) } );
            }
        }
        // time-normalised distance
        DTW_matrix(n-1,m-1) = DTW_matrix(n-1,m-1) / (n);
    }

    // auto end_timer = sc.now();
    // cout << static_cast<duration<double>>(end_timer - start_timer).count() << '\n';

    // backtrace to find path
    i = DTW_matrix.rows() - 2;
    j = DTW_matrix.cols() - 2;
 
    ArrayXXd path(n,m);
    vector<vector<int>> path;
    
    path.resize(n);
    list<list<int>> path(n,m);
    int min_ind, cnt = 0;
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
   //  for( unsigned int x=0; x<path.size(); x++)
   //  {
   //  	for( unsigned int y=0; y<path[x].size(); y++)
   //  	{	
   //  		cout << path[x][y] << " ";
   //  	}
   //  	cout << endl;
   //  }
   
    return DTW_matrix(n-1,m-1);
}


