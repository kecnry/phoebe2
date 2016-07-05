#if !defined(__rot_star_h)
#define __rot_star_h

/*
  Library dealing with rotating stars defined by the potential 

  Omega = 1/sqrt(x^2 + y^2 + z^2) + 1/2 omega^2 (x^2+ y^2) 

  Author: Martin Horvat,  June 2016 
*/

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

// general routines
#include "utils/utils.h"                  // Misc routines (sqr, solving poly eq,..)
#include "triang/triang_marching.h"       // Maching triangulation
#include "eclipsing/eclipsing.h"          // Eclipsing/Hidden surface removal
#include "triang/bodies.h"                // Definitions of different potentials

namespace rot_star {
  
  /*
    "Lagrange point" of the rotating star potential 
      
        (L_1, 0, 0)     (-L_1, 0, 0)
  
    Input:
      omega - parameter of the potential
      
    Return 
      value Lagrange point L_1: if (omega == 0) return NaN
    
  */ 
  template <class T>
  T lagrange_point(const T & omega) {
    if (omega == 0) return std::nan("");
    return std::cbrt(omega*omega);
  }
  
  
  /*  
    Potential on the x-axis
    
    Input:
      x - position on the x-axis
      omega - parameter of the potential
    
    Return 
      value of the potential
  */
 
  template<class T> 
  T potential_on_x_axis( const T & x, const T & omega) {
    return 1/std::abs(x) + omega*omega*x*x/2;
  }
  
  
   /*
    Points of on the x-axis satisfying
    
      Omega(x,0,0) = Omega_0
    
    The Roche lobes are limited by these points.
    
    Input:
      Omega0 - reference value of the potential
      omega - parameter of the potential
            
      trimming: 
        if false : all solutions
        if true  : trims solutions which bounds Roche lobes,
                   there are even number bounds
        
    Output:
      p - x-values of the points on x-axis sorted
  */
  template<class T> 
  void points_on_x_axis(
    std::vector<T> & points,
    const T & Omega0,
    const T & omega,
    const bool & trimming = true){
    
    if (omega == 0) return 1/Omega0;
    
    std::vector<T> roots;
    
    T a[4] = {1, -Omega0, 0, omega*omega/2};
    
    utils::solve_cubic(a, roots);
    
    for (auto && x: roots)
      if (x > 0) {
          points.push_back(x);
          points.push_back(-x);
      }
    
    std::sort(points.begin(), points.end());
    
    if (points.size() >= 2 && trimming) {
      points.pop_back();
      points.erase(points.begin());
    }
  }
  
  /*
    Pole of the star i.e.  smallest z > 0 such that
  
      Omega(0,0,z) = Omega0
    
    Input:
      Omega0 - value of potential
      omega - parameter of the potential connected to rotation
      
    Return:
      height of pole > 0
    
    Comment:
      if pole is not found it return -1
  */
  template <class T>
  T pole(const T & Omega0, const T & omega) {
    return 1/Omega0; 
  }
  
  
  /*
    Returning the critical values of the star potential.
    
    Input:
      omega - potential parameter
    
    Return:
      critical value of the potential : if (omega == 0) return NaN
  */

  template<class T> 
  T critical_potential(const T & omega) {
    if (omega == 0) return std::nan("");
    //return potential_on_x_axis(lagrange_point(omega), omega);
    return 3*std::pow(omega, 2./3)/2;
  }
  
 /*
    Computing area of the surface and the volume of the rotating star 
    lobe.
  
    Input:
      omega - parameter of the potential
      Omega0 - value of the potential 
      choice -
        1  - area
        2  - volume
        3  - both

    Using: Gauss-Lagrange integration along z direction
       
    Output:
      av[2] = {area, volume}
    
    Precision is at least 10^-5.
    
    Ref: 
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better  
  */
 
  template<class T> 
  void area_volume(
    double *av, 
    const unsigned &choice, 
    const T & Omega0, 
    const T & omega) {

    //
    // What is calculated
    //
    
    bool 
      b_area = (choice & 1u) == 1u,
      b_volume = (choice & 2u) == 2u;

    if (!b_area && !b_volume) return;
    
    T Omega2 = Omega0*Omega0,
      Omega3 = Omega0*Omega2;
      
    if (omega == 0) {
      
      if (b_area) av[0] = utils::M_4PI/Omega2;
      if (b_volume) av[1] = utils::M_4PI/(3*Omega3); 
        
      return;
    }  
    
    //
    // Testing if the solution exists
    //  
    
    T t = omega*omega/Omega3; // t in [t_crit], t_crit = 8/27
    
    if (t > 8./27) {   // critical value
      std::cerr << "rotstar::area_volume:There is no solution for equator.\n";
      return;
    }

    //
    // Analytic approximation (generated in rot_star.nb)
    // relative precision at least 1e-5 for t < 0.1
    
    if (t < 0.1) {
      
      T f = utils::M_4PI/Omega2;
      
      if (b_area) {
        const T a[] = {1.,0.6666666666666666,1.,1.9428571428571428,4.314285714285714,10.398268398268398,26.48877788877789,70.22541902541903,191.8665770657039,536.7383091809828,1536.0162254282043};
              
        T A = a[0] + t*(a[1] + t*(a[2] + t*(a[3] + t*(a[4] + t*(a[5] + t*(a[6] + t*(a[7] + t*(a[8] + t*(a[9] + t*a[10])))))))));
        
        av[0] = f*A;
      }
      
      if (b_volume) {
        const T a[] = {1.,1.,1.6,3.142857142857143,6.933333333333334,16.484848484848484,41.302697302697304,107.56923076923077,288.6243489583333,793.03125,2230.111036424513};
        
        T V = a[0] + t*(a[1] + t*(a[2] + t*(a[3] + t*(a[4] + t*(a[5] + t*(a[6] + t*(a[7] + t*(a[8] + t*(a[9] + t*a[10])))))))));
        
        av[1] = f*V/(3*Omega0);
      }
      
      return;
    }
    
    //
    // Integrate using new variables
    //  v = Omega z, u = Omega rho  rho^2 = x^2 + y^2 
    // we additionally introduce
    //  s = u^2,
    // and rewrite equation for potential:
    //  1 = 1/sqrt(s + v^2) + 1/2 t s     v in [0,1]   
    // where
    //  v=1 is pole, v=0 is equator
    // with t = omega^2/Omega^3
    
    // u du/dv = 1/2 ds/dv
    
    const int m = 1 << 16;
        
    T dv = 1./m, v = 1,
      s = 0, A = 0, V = 0, k[4][3],
      v1, s1, q, F; // auxiliary variables  
    
    
    // FIX: just that I don't get may be used uninitialized in this function 
    for (int i = 0; i < 4; ++i) k[i][0] = k[i][1] = k[i][2] = 0;
  
    for (int i = 0; i < m; ++i) {
      
      //
      // 1. step
      //
      v1 = v; 
      s1 = s; 
      q = v1*v1 + s1; 
      F = 2*v1/(1 - t*q*std::sqrt(q));              // = ds/dv
      
      k[0][0] = dv*F;                                 // = dv*ds/dv
      if (b_area) k[0][1] = dv*std::sqrt(s1 + F*F/4); // dv (u^2 + (udu/dv)^2)^(1/2)
      if (b_volume) k[0][2] = dv*s1;                  // dv u^2
        
      // prepare: y1 = y + k0/2
      s1 = s + k[0][0]/2;
      
      //
      // 2. step 
      //
      v1 = v - dv/2;
      q = v1*v1 + s1; 
      F = 2*v1/(1 - t*q*std::sqrt(q));               // =ds/dv
       
      k[1][0] = dv*F;                                  // = dv*ds/dv
      if (b_area) k[1][1] = dv*std::sqrt(s1 + F*F/4);  // dv (u^2 + (udu/dv)^2)^(1/2)
      if (b_volume) k[1][2] = dv*s1;                   // dv u^2
      
      // prepare: y1 = y + k1/2
      s1 = s + k[1][0]/2;
      
      //
      // 3. step
      //
      q = v1*v1 + s1; 
      F = 2*v1/(1 - t*q*std::sqrt(q));                // =ds/dv
      
      k[2][0] = dv*F;                                   // = dv*ds/dv
      if (b_area) k[2][1] = dv*std::sqrt(s1 + F*F/4);   // dv (u^2 + (udu/dv)^2)^(1/2)
      if (b_volume) k[2][2] = dv*s1;                    // dv u^2
      
      // prepare: y1 = y + k1/2
      s1 = s + k[2][0];
      
      // 4. step
      v1 = v - dv;
      q = v1*v1 + s1; 
      F = 2*v1/(1 - t*q*std::sqrt(q));                // =ds/dv
      
      k[3][0] = dv*F;                                   // = dv*ds/dv
      if (b_area) k[3][1] = dv*std::sqrt(s1 + F*F/4);   // dv (u^2 + (udu/dv)^2)^(1/2)
      if (b_volume) k[3][2] = dv*s1;                    // dv u^2
    
      s += (k[0][0] + 2*(k[1][0] + k[2][0]) + k[3][0])/6;
      if (b_area) A += (k[0][1] + 2*(k[1][1] + k[2][1]) + k[3][1])/6;
      if (b_volume) V += (k[0][2] + 2*(k[1][2] + k[2][2]) + k[3][2])/6;
          
      v -= dv;
    }
    
    if (b_area) av[0] = utils::M_4PI*A/Omega2;
    if (b_volume) av[1] = utils::M_2PI*V/Omega3;
  }
  
  /*
    Computing the volume of the rotating star and its derivatives w.r.t. to Omega, ...
    
    The range on x-axis is [x0, x1].
  
    Input:
      Omega0 - value of the potential
      omega - parameter of th potential
      choice : composing from mask
        1  - Volume, stored in av[0]
        2  - dVolume/dOmega, stored in av[1]
                 
    Using: Integrating surface in cylindric geometry
      a. Gauss-Lagrange integration in phi direction
      b. RK4 in x direction
    
    Precision:
      At the default setup the relative precision should be better 
      than 1e-5.
             
    Output:
      res = {Volume, dVolume/dOmega, ...}
  
    Ref: 
      * https://en.wikipedia.org/wiki/Gaussian_quadrature
      * https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula
      * http://mathworld.wolfram.com/LobattoQuadrature.html <-- this would be better  
  */
  
  template<class T> 
  void volume(
    T *res,
    const unsigned & choice,
    const T & Omega0,
    const T & omega)
  {
  
    //
    // What is calculated
    //
    
    bool 
      b_Vol= (choice & 1u) == 1u,
      b_dVoldOmega = (choice & 2u) == 2u;

    if (!b_Vol && !b_dVoldOmega) return;
    
    T Omega2 = Omega0*Omega0,
      Omega3 = Omega0*Omega2;
      
    if (omega == 0) {
      
      T Vol =  utils::M_4PI/(3*Omega3);
      
      if (b_Vol) res[0] = Vol;
      if (b_dVoldOmega) res[1] = -3*Vol/Omega0; 
        
      return;
    }  

    //
    // Testing if the solution exists
    //
    
    T t = omega*omega/Omega3; // t in [t_crit], t_crit = 8/27
    
    if (t > 8./27) {   // critical value
      std::cerr << "rotstar::area_volume:There is no solution for equator.\n";
      return;
    }

    //
    // Analytic approximation (generated in rot_star.nb)
    // relative precision at least 1e-5 for t < 0.1
    if (t < 0.1) {
      
      const T a[] = {
        1., 1.,1.6,3.142857142857143,6.933333333333334,
        16.484848484848484,41.302697302697304,107.56923076923077,
        288.6243489583333,793.03125,2230.111036424513};
      
      T Vol = 
          a[0] + t*(a[1] + t*(a[2] + t*(a[3] + t*(a[4] + t*(a[5] + 
          t*(a[6] + t*(a[7] + t*(a[8] + t*(a[9] + t*a[10]))))))))),
        f = utils::M_4PI/(3*Omega3);
      
      if (b_Vol) res[0] = f*Vol;
      
      if (b_dVoldOmega) {
        T dVoldt = 
            a[1] + t*(2*a[2] + t*(3*a[3] + t*(4*a[4] + t*(5*a[5] + 
            t*(6*a[6] + t*(7*a[7] + t*(8*a[8] + t*(9*a[9] + t*10*a[10]))))))));
        
        res[1] = -3*f*(Vol + t*dVoldt)/Omega0;
      }
      
      return;
    }
    
    //
    // Integrate using new variables
    //  v = Omega z, u = Omega rho  rho^2 = x^2 + y^2 
    // we additionally introduce
    //  s = u^2,
    // and rewrite equation for potential:
    //  1 = 1/sqrt(s + v^2) + 1/2 t s     v in [0,1]   
    // where
    //  v=1 is pole, v=0 is equator
    // with t = omega^2/Omega^3
    
    // u du/dv = 1/2 ds/dv
    
    const int m = 1 << 16;
        
    T dv = 1./m, v = 1,
      s = 0, Vol = 0, dVoldOmega = 0, k[4][3],
      v1, s1, q, f, F; // auxiliary variables  
    
    
    // FIX: just that I don't get may be used uninitialized in this function 
    for (int i = 0; i < 4; ++i) k[i][0] = k[i][1] = k[i][2] = 0;
  
    for (int i = 0; i < m; ++i) {
      
      //
      // 1. step
      //
      v1 = v; 
      s1 = s; 
      q = v1*v1 + s1;
      f = q*std::sqrt(q);
      F = 2*v1/(1 - t*f);              // = ds/dv
      
      k[0][0] = dv*F;                  // = dv*ds/dv
      if (b_Vol) k[0][1] = dv*s1;                             // dv u^2
      if (b_dVoldOmega) k[0][2] = dv*(s1 + t*s1*f/(1 - t*f)); // dv (s + t ds/dt)
        
      // prepare: y1 = y + k0/2
      s1 = s + k[0][0]/2;
      
      //
      // 2. step 
      //
      v1 = v - dv/2;
      q = v1*v1 + s1; 
      f = q*std::sqrt(q);
      F = 2*v1/(1 - t*f);               // = ds/dv
       
      k[1][0] = dv*F;                   // = dv*ds/dv
      if (b_Vol) k[1][1] = dv*s1;                             // dv u^2
      if (b_dVoldOmega) k[1][2] = dv*(s1 + t*s1*f/(1 - t*f)); // dv (s + t ds/dt)
       
      // prepare: y1 = y + k1/2
      s1 = s + k[1][0]/2;
      
      //
      // 3. step
      //
      q = v1*v1 + s1; 
      f = q*std::sqrt(q);
      F = 2*v1/(1 - t*f);               // = ds/dv
      
      k[2][0] = dv*F;                   // = dv*ds/dv
      if (b_Vol) k[2][1] = dv*s1;                             // dv u^2
      if (b_dVoldOmega) k[2][2] = dv*(s1 + t*s1*f/(1 - t*f)); // dv (s + t ds/dt)
   
      // prepare: y1 = y + k1/2
      s1 = s + k[2][0];
      
      // 4. step
      v1 = v - dv;
      q = v1*v1 + s1; 
      f = q*std::sqrt(q);
      F = 2*v1/(1 - t*f);               // = ds/dv
      
      k[3][0] = dv*F;                   // = dv*ds/dv
      if (b_Vol) k[3][1] = dv*s1;                             // dv u^2
      if (b_dVoldOmega) k[3][2] = dv*(s1 + t*s1*f/(1 - t*f)); // dv (s + t ds/dt)
      
      s += (k[0][0] + 2*(k[1][0] + k[2][0]) + k[3][0])/6;
      if (b_Vol) Vol += (k[0][1] + 2*(k[1][1] + k[2][1]) + k[3][1])/6;
      if (b_dVoldOmega) dVoldOmega += (k[0][2] + 2*(k[1][2] + k[2][2]) + k[3][2])/6;
          
      v -= dv;
    }
    
    f = utils::M_2PI/Omega3;
    
    if (b_Vol) res[0] = f*Vol;
    if (b_dVoldOmega) res[1] = -3*f*dVoldOmega/Omega0;  
  }
  
} // namespace rot_star

#endif // #if !defined(__rot_star_h) 
