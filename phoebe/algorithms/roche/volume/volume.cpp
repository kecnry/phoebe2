#include <iostream>
#include <cmath>

#include "volume.h"
#include "../gen_roche.h" 

int main(){
  #if 0
  //
  // overcontact case
  //
  double 
    q = 0.5,
    F = 0.5,
    deltaR = 1,
    Omega0 = 2.65;
   #endif
  
  
  #if 1
  //
  // Phoebe generic case: detached case
  //
  double
    q = 1,
    F = 1,
    deltaR = 1,
    Omega0 = 10;
   #endif
   
  
  std::cout.precision(16);
  
  std::vector<double> x_points;
  
  gen_roche::points_on_x_axis(x_points, Omega0, q, F, deltaR);
  
  std::cout << "x0=" << x_points[0] << " x1=" << x_points[1] << '\n';
  
  double
    xrange[2] = {x_points[0], x_points[1]},
    av[2]; 
     
  gen_roche::area_volume(av, xrange, q, F, deltaR);
  
  std::cout << "A=" << av[0] << " V=" << av[1] << '\n';
  
  
  return 0;
}