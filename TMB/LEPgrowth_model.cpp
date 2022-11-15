// LEP growth model
/*
 * LEP growth model using both age and length data and tag-increment data
 */

#include <TMB.hpp>

template <class Type> 
void richards_growth(vector<Type> ages, Type k, Type L_inf, Type t_0, Type p, vector<Type>& pred_lengths) {
  int n_samples = ages.size();
  for(int i = 0; i < n_samples; ++i) {
    pred_lengths(i) = L_inf * pow(1 + 1 / p * exp(-k * (ages(i) - t_0)),-p);
  }
}
// transform Y -Inf-Inf -> X bound lb - ub
template <class Type> 
Type invlogit_general(Type& Y, Type& lb, Type& ub) {
  return(lb + (ub - lb) * (1 / (1 + exp(-Y))));
}


template<class Type>
Type objective_function<Type>::operator() ()
{
  /*
   * Declare Namespace
   */
  using namespace density;

  // Input parameters
  // model dimensions
  DATA_VECTOR(ages_from_age_length);                // ages can be non-integer. length = n_age_length_samples
  DATA_VECTOR(lengths_from_age_length);             // lengths can be non-integer. length = n_age_length_samples
  int n_age_length_samples = ages_from_age_length.size();
  
  
  DATA_VECTOR(lengths_at_release);                 // length at release. length = n_tag_increment_samples
  DATA_VECTOR(lengths_at_recovery);                // length at recovery. length = n_tag_increment_samples
  DATA_VECTOR(time_at_liberty);                    // time-at-liberty expressed as years.length = n_tag_increment_samples
  int n_tag_increment_samples = lengths_at_release.size();
  DATA_VECTOR(ages_for_report);                    // ages we want to plot and reprot mean length at-age estimates
  
  
  DATA_VECTOR(p_bounds);                           // upper and lower bound for the P parameter
  DATA_VECTOR(t0_bounds);                          // upper and lower bound for the t0 parameter
  
  // Estimable parameters
  PARAMETER(ln_cv_length_at_age);                   // log CV for length at age distribution
  PARAMETER(ln_k);                                  // log growth coefficient
  PARAMETER(logit_t0);                              // logistic corresponds to an inflexion point on the cur
  PARAMETER(ln_L_inf);                              // log the asymptotic lengthn
  PARAMETER(logit_p);                               // logistic of the shape parameter
  PARAMETER(ln_cv_length_release);                  // log standard deviation for the age-length observation
  PARAMETER(ln_cv_length_recovery);                 // log standard deviation for the age-length observation
  PARAMETER_VECTOR(ln_age_at_release);                      // log of the age of each fish released. length = n_tag_increment_samples
  
  // Hyper distributions
  PARAMETER(ln_mu_age_release);
  PARAMETER(ln_sd_age_release);
  
  /*
   *  Transform parameters
   */
  Type cv_length_at_age = exp(ln_cv_length_at_age);
  Type k = exp(ln_k);
  Type t0 = invlogit_general(logit_t0, t0_bounds(0), t0_bounds(1));
  Type L_inf = exp(ln_L_inf);
  Type p = invlogit_general(logit_p, p_bounds(0), p_bounds(1));
  Type cv_length_release = exp(ln_cv_length_release);
  Type cv_length_recovery = exp(ln_cv_length_recovery);
  vector<Type> age_at_release = exp(ln_age_at_release);

  Type mu_age_release = exp(ln_mu_age_release);
  Type sd_age_release = exp(ln_sd_age_release);
  
  vector<Type> nll(4);
  nll.setZero();
  /*
   * 0 = age-length log likelihood
   * 1 = length at release log-likelihood
   * 2 = length at recapture log-likelihood
   * 3 = age_at_release hyper distribution
   */
  
  
  /*
   *  Generate expected/predicted values
   */  
  vector<Type> pred_lengths_at_release(n_tag_increment_samples);
  vector<Type> pred_lengths_at_recovery(n_tag_increment_samples);
  vector<Type> pred_lengths_length_at_age(n_age_length_samples);
  vector<Type> mean_length_at_age(ages_for_report.size());
  
  vector<Type> age_at_recovery = age_at_release + time_at_liberty;
  
  richards_growth(ages_from_age_length, k, L_inf, t0, p ,pred_lengths_length_at_age);
  richards_growth(age_at_release, k, L_inf, t0, p ,pred_lengths_at_release);
  richards_growth(age_at_recovery, k, L_inf, t0, p ,pred_lengths_at_recovery);
  richards_growth(ages_for_report, k, L_inf, t0, p ,mean_length_at_age);
  
  // Calculate the log-likelhood
  nll(0) -= sum(dnorm(lengths_from_age_length, pred_lengths_length_at_age, pred_lengths_length_at_age * cv_length_at_age, 1));
  nll(1) -= sum(dnorm(lengths_at_release, pred_lengths_at_release, pred_lengths_at_release * cv_length_release, 1));
  nll(2) -= sum(dnorm(lengths_at_recovery, pred_lengths_at_recovery, pred_lengths_at_recovery * cv_length_recovery, 1));
  // Add hyper prior for age at release
  nll(3) -= sum(dnorm(ln_age_at_release, ln_mu_age_release, sd_age_release, 1));
  /*
   * Report
   */
  REPORT(cv_length_at_age);                   
  REPORT(k);                                 
  REPORT(t0);                                
  REPORT(L_inf);                            
  REPORT(p);                                  
  REPORT(cv_length_release);                 
  REPORT(cv_length_recovery);                
  REPORT(age_at_release);                     
  REPORT(mu_age_release);
  REPORT(sd_age_release);
  
  REPORT(pred_lengths_at_release);
  REPORT(pred_lengths_at_recovery);
  REPORT(pred_lengths_length_at_age);
  REPORT(mean_length_at_age);
  REPORT(nll);
  
  // get standard errors from these parameters
  ADREPORT(cv_length_at_age);                   
  ADREPORT(k);                                 
  ADREPORT(t0);                                
  ADREPORT(L_inf);                            
  ADREPORT(p);                                  
  ADREPORT(cv_length_release);                 
  ADREPORT(cv_length_recovery);                
  ADREPORT(age_at_release);                     
  ADREPORT(mu_age_release);
  ADREPORT(sd_age_release);
  ADREPORT(mean_length_at_age);
  return nll.sum();
}

