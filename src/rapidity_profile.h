#ifndef ETA_H
#define ETA_H

#include <cmath>
#include <vector>
#include <gsl/gsl_fft_complex.h>
#define REAL(z,i) ((z)[2*(i)])
#define IMAG(z,i) ((z)[2*(i)+1])

double const sqrt2 = std::sqrt(2);
constexpr double TINY = 1e-12;

///----------------------------------------------------------------------------------------------///
/// feel free to try you own parametrization of y-mean. y-std and y-skew as function of Ta and Tb///
///----------------------------------------------------------------------------------------------///
double inline mean_function(double ta, double tb, double exp_ybeam){
  if (ta < TINY && tb < TINY) return 0.;
  return 0.5 * std::log((ta*exp_ybeam + tb/exp_ybeam) / std::max(ta/exp_ybeam + tb*exp_ybeam, TINY));
}

double inline std_function(double ta, double tb){
  (void)ta;
  (void)tb;
  return 1.;
}

double inline skew_function(double ta, double tb){
  return (ta - tb)/std::max(ta + tb, TINY);
}


/// A fast pseudorapidity to rapidity transformer using pretabulated values
/// Output both y(eta) and dy/deta(eta)
class fast_eta2y {
 private:
  double etamax_;
  double deta_;
  std::size_t neta_;
  std::vector<double> y_;
  std::vector<double> dydeta_;
 public:
  fast_eta2y(double J, double etamax, double deta)
      : etamax_(etamax),
        deta_(deta),
        neta_(std::ceil(2.*etamax_/deta_)+1),
        y_(neta_, 0.),
        dydeta_(neta_, 0.) {

    for (std::size_t ieta = 0; ieta < neta_; ++ieta) {
      double eta = -etamax_ + ieta*deta_;
      double Jsh = J*std::sinh(eta);
      double sq = std::sqrt(1. + Jsh*Jsh);
      y_[ieta] = std::log(sq + Jsh);
      dydeta_[ieta] = J*std::cosh(eta)/sq;
    }
  }
  
  double rapidity(double eta){
    double steps = (eta + etamax_)/deta_;
    double xi = std::fmod(steps, 1.);
    std::size_t index = std::floor(steps);
    return y_[index]*(1. - xi) + y_[index+1]*xi;
  }

  double Jacobian(double eta){
    double steps = (eta + etamax_)/deta_;
    double xi = std::fmod(steps, 1.);
    std::size_t index = std::floor(steps);
    return dydeta_[index]*(1. - xi) + dydeta_[index+1]*xi;
  }

};

class cumulant_generating{
private:
	size_t const N;
	double * data, * dsdy;
	double eta_max;
	double deta;
	double center;
	
public:
	cumulant_generating(): N(256), data(new double[2*N]), dsdy(new double[2*N]){};
	void calculate_dsdy(double mean, double std, double skew){
		double k1, k2, k3, amp, arg;
		// adaptive eta_max = 3.33*std;
		center=mean;
		eta_max = std*3.33;
		deta = 2.*eta_max/(N-1.);
		double fftmean = eta_max/std;
    	for(size_t i=0;i<N;i++)
   		{
        	k1 = M_PI*(i-N/2.0)/eta_max*std;
			k2 = k1*k1;
			k3 = k2*k1;

        	amp = std::exp(-k2/2.0);
        	arg = fftmean*k1+skew/6.0*k3*amp;
        
			REAL(data,i) = amp*std::cos(arg);
        	IMAG(data,i) = amp*std::sin(arg);
    	}
   		gsl_fft_complex_radix2_forward(data, 1, N);
    
    	for(size_t i=0;i<N;i++)
    	{
        	dsdy[i] = REAL(data,i)*(2.0*static_cast<double>(i%2 == 0)-1.0);
    	}
	}
	double interp_dsdy(double y){
		y = y-center;
		if (y < -eta_max || y >= eta_max) return 0.0;
		double xy = (y+eta_max)/deta;
		size_t iy = std::floor(xy);
		double ry = xy-iy;
		return dsdy[iy]*(1.-ry) + dsdy[iy+1]*ry;
	}
};
#endif
