#include <TMB.hpp>
/*
 * isNA
 */
template<class Type>
bool isNA(Type x){
  return R_IsNA(asDouble(x));
}
/* Parameter transform */
template <class Type>
Type f(Type x){return Type(2)/(Type(1) + exp(-Type(2) * x)) - Type(1);}
/*
 * 
 */
template<class Type>
Type posfun(Type x, Type eps, Type &pen){
  pen += CppAD::CondExpLt(x,eps,Type(0.01)*pow(x-eps,2),Type(0));
  return CppAD::CondExpGe(x,eps,x,eps/(Type(2)-x/eps));
}
/*
 * 
 */
template <class Type> 
Type square(Type x){return x*x;}

template <class Type> 
vector<Type> square(vector<Type>& x) {
  return x*x;
}
// transform Y -Inf-Inf -> X bound lb - ub
template <class Type> 
Type invlogit_general(Type& Y, Type& lb, Type& ub) {
  return(lb + (ub - lb) * (1 / (1 + exp(-Y))));
}


// logistic ogive function
template <class Type> 
vector<Type> logistic_ogive(vector<Type> ages, Type sel_50, Type sel_95) {
  std::cout << "logistic_ogive\n";
  int n_ages = ages.size();
  vector<Type> logis(n_ages);
  for (int age = 0;  age < n_ages; ++age) {
    logis[age] = Type(1.0) / (Type(1.0) + pow(Type(19.0), (sel_50 - ages[age]) / sel_95));
  }
  return logis;
}
/*
 * Geometric mean
 */
template <class Type> 
Type geo_mean(vector<Type>& x){
  return exp((log(x).sum())/x.size());
}

template <class Type>
void get_covar(vector<Type> theta, matrix<Type>& covar, int n, int type) {
  // https://github.com/glmmTMB/glmmTMB/blob/9cb23d77afc042be5bced3713c40529023d9ef7e/glmmTMB/src/glmmTMB.cpp#L361
  if (type == 1) {
    // case: diag_covstruct
    //std::cerr << "diag_covstruct" << std::endl;
    Type sd = exp(theta(0));
    for(int i = 0; i < n; ++i)
      covar(i,i) = sd * sd;
  } else if (type == 2) {
    // case: us_covstruct
    //std::cerr << "us_covstruct" << std::endl;
    vector<Type> sd = exp(theta.head(n));
    vector<Type> corr_transf = theta.tail(theta.size() - n);
    density::UNSTRUCTURED_CORR_t<Type> nldens(corr_transf);
    // for covariance
    //density::VECSCALE_t<density::UNSTRUCTURED_CORR_t<Type> > scnldens = density::VECSCALE(nldens, sd);
    for(int i=0; i<n; i++) {
      for(int j=0; j<n; j++) {
        covar(i,j) = nldens.cov()(i,j) * sd(i) * sd(j);
      }
    }
  } else if (type == 3){
    // case: cs_covstruct Compound Symmetry: Heterogenous.
    //std::cerr << "cs_covstruct" << std::endl;
    
    vector<Type> sd = exp(theta.head(n));
    Type corr_transf = theta(n);
    Type a = Type(1) / (n - Type(1)); // Correcting for violations of sphericity 
    Type rho = invlogit(corr_transf) * (Type(1) + a) - a;
    for(int i=0; i<n; i++)
      for(int j=0; j<n; j++)
        covar(i,j) = (i==j ? Type(1) * sd(i) * sd(i) : rho * sd(i) * sd(j)) ;
    
  } else if (type == 4){
    // case: toep_covstruct Toeplitz: Heterogenous.
    //std::cerr << "toep_covstruct" << std::endl;
    vector<Type> sd = exp(theta.head(n));
    vector<Type> parms = theta.tail(n-1);              // Corr parms
    parms = parms / sqrt(Type(1.0) + parms * parms );  // Now in (-1,1)
    for(int i=0; i<n; i++)
      for(int j=0; j<n; j++)
        covar(i,j) = (i==j ? Type(1) * sd(i) * sd(i) : parms( (i > j ? i-j : j-i) - 1 ) * sd(i) * sd(j));
  } else if (type == 5){
    //std::cerr << "ar1_covstruct" << std::endl;
    // case: ar1_covstruct
    //  * NOTE: Valid parameter space is phi in [-1, 1]
    //  * NOTE: 'times' not used as we assume unit distance between consecutive time points.
    vector<Type> sd(n);
    sd.fill(exp(theta(0)));
    Type corr_transf = theta(1);
    Type phi = corr_transf / sqrt(1.0 + pow(corr_transf, 2));
    for(int i=0; i < n; i++){
      for(int j=0; j < n; j++){
        covar(i,j) = pow(phi, abs(i-j)) * sd(i) * sd(i);
      }
    }
  }
  
  return;
}
/*
 * centred log transform
 */
template <class Type> 
vector<Type> crl(vector<Type>& x) { 
  return log(x / geo_mean(x));
}
/*
 * rmultinomm - for simulate call
 */
template <class Type> 
vector<Type> rmultinom(vector<Type> prob, Type N) { 
  vector<Type> sim_X(prob.size());
  sim_X.setZero();
  // Now simulate using the uniform random variable
  Type rng_uniform;
  Type cumulative_expect;
  while(N > 0) {
    rng_uniform = runif(Type(0),Type(1));
    //std::cout << rng_uniform << " ";
    cumulative_expect = 0.0;
    for (unsigned i = 0; i < prob.size(); ++i) {
      cumulative_expect += prob[i];
      if (cumulative_expect >= rng_uniform) {
        sim_X[i] += 1.0;
        break;
      }
      //sim_X[prob.size() - 1] += 1.0;
    }
    N -= 1;
  }
  //std::cout << "\n";
  return(sim_X);
}

/*
 *  Simulate a single draw from a multinomial-dirichlet distribution
 */
template <class Type> 
vector<Type> rdirichletmulti(vector<Type> fitted_props, Type& n_eff, Type& theta) {
  vector<Type> dirichlet_draw(fitted_props.size()); 
  for(int ndx = 0; ndx < fitted_props.size(); ndx++) 
    dirichlet_draw(ndx) = rgamma(fitted_props(ndx) * theta * n_eff, (Type)1.0);// shape, rate = 1.0
  
  Type dirich_total = dirichlet_draw.sum();
  dirichlet_draw /= dirich_total;
  return(rmultinom(dirichlet_draw, n_eff));
}
/*
 * inverse centred log transform
 * Up to a constant so standardise
 */
template <class Type> 
vector<Type> inv_crl(vector<Type>& y){ 
  return exp(y) / (exp(y)).sum();
}

// Beverton-Holt SR relationship function
template<class Type>
Type BevertonHolt(Type SSB, Type B0, Type h) {
  Type ssb_ratio = SSB / B0;
  Type part_2 = (1 - ((5*h - 1) / (4*h)) * ( 1 - ssb_ratio));
  return (ssb_ratio / part_2);
}

// Beverton-Holt SR relationship function without equilibrium assumptions
template<class Type>
Type BevertonHoltNoEquil(Type a, Type b, Type SSB) {
  Type Rt = (a + SSB) / (SSB * b);
  return Rt;
}



/*
 * A simple age structure stock assessment in TMB, that has two fisheries
 * 
 */

template<class Type>
Type objective_function<Type>::operator() () {
  /*
   * Declare Namespace
   */
  using namespace density;
  // model dimensions
  DATA_VECTOR(ages);    // assumes min(ages) >= 1
  DATA_VECTOR(years);   // annual years
  int nyears = years.size();
  int nages = ages.size();
  DATA_INTEGER(maxAgePlusGroup);  // 1 = yes, 0 = no
  
  // Observation info
  DATA_MATRIX(ageing_error_matrix);     // nages * nages
  DATA_IVECTOR(survey_year_indicator);  // 1 = calculate, 0 = ignore nyears
  DATA_VECTOR(survey_obs);              // Relative index (this will be ignored if survey_comp_type = 1)
  DATA_VECTOR(survey_cv);               // CV for relative index (this will be ignored if survey_comp_type = 1)
  DATA_VECTOR_INDICATOR(survey_index_keep, survey_obs);  // For one-step predictions
  
  DATA_ARRAY(survey_trans_comp);        // ALN transformed observations dim = length()survey_min_age:survey_max_age) - 1  NB x sum(survey_year_indicator) 
  // fishery_comp_type = 1, log(N), fishery_comp_type = 0 & fishery_comp_likelihood = 0 alr(N), fishery_comp_likelihood = 1 N_eff * prop(N)
  DATA_ARRAY_INDICATOR(keep_survey_comp, survey_trans_comp);
  DATA_SCALAR(survey_min_age);         // survey_age_indicator.size() = nages, 
  DATA_SCALAR(survey_max_age);          // survey_age_indicator.size() = nages, 
  DATA_INTEGER(survey_age_plus_group);  // 0 no plus group, 1 = an accumlation group
  DATA_INTEGER(survey_comp_type);       // 1 = numbers at age, 0 = proportions at age;
  DATA_INTEGER(survey_covar_structure); // 0 = iid, 5 = AR(1), 2 = Unstructured.
  DATA_INTEGER(survey_comp_likelihood); // 0 = MVN (can be applied to both comp_type), 1 = Multinomial, 2 = dirichlet-multinomial
  
  DATA_IVECTOR(fishery_year_indicator);  // 1 = calculate, 0 = ignore
  DATA_SCALAR(fishery_min_age);          // survey_age_indicator.size() = nages, 
  DATA_SCALAR(fishery_max_age);          // survey_age_indicator.size() = nages, 
  DATA_INTEGER(fishery_age_plus_group);  // 0 no plus group, 1 = an accumlation group
  DATA_ARRAY(fishery_trans_comp);        // sum(fishery_year_indicator) x fishery_nages NB!!!! different dims to most other matrices..
  // fishery_comp_type = 1, log(N), fishery_comp_type = 0 & fishery_comp_likelihood = 0 alr(N), fishery_comp_likelihood = 1 N_eff * prop(N)
  DATA_INTEGER(fishery_comp_type);       
  // numbers at age = 1, proportions at age = 0
  DATA_INTEGER(fishery_comp_likelihood); // 0 = MVN (can be applied to both comp_type), 1 = Multinomial, 2 = dirichlet-multinomial
  DATA_INTEGER(fishery_covar_structure); // 1 = iid (1), 5 = AR(1) (2), 2 = Unstructured (A_f(A_f - 1)/2).
  DATA_ARRAY_INDICATOR(keep_fishery_comp, fishery_trans_comp);
  
  DATA_IVECTOR(ycs_estimated);    // 1 = estimated, 0 = ignore
  DATA_INTEGER(standardise_ycs);  // 1 = yes, 0 = No
  
  // Catch info
  DATA_VECTOR(catches);               // length = nyears
  DATA_VECTOR(propZ_ssb);             // proportion of Z for SSB, length nyears
  DATA_VECTOR(propZ_survey);          // proportion of Z for SSB, length nyears
  // Biological info
  DATA_ARRAY(propMat);                // Proportion Mature (used in SSB)    dim = nages x nyears
  DATA_ARRAY(stockMeanLength);        // Stock Mean weight used in SSB + survey calculation  dim = nages x nyears
  DATA_ARRAY(catchMeanLength);        // Stock Mean weight used in Catch equation calculation dim = nages x nyears
  DATA_SCALAR(natMor);                // Instantaneous natural mortality        
  DATA_SCALAR(steepness);             // Instantaneous natural mortality     
  DATA_SCALAR(mean_weight_a);         // a parameter mean weight from length 
  DATA_SCALAR(mean_weight_b);         // b parameter mean weight from length    
  
  DATA_INTEGER(stockRecruitmentModelCode); // SR relationship 0 = RW, 1 = Ricker, 2 = BH
  
  // Bounds for selectivity parameters for model stability
  DATA_VECTOR(sel_ato95_bounds);  // length 2
  DATA_VECTOR(sel_a50_bounds);    // length 2
  
  
  /*
   * Parameters to estiamte.
   */
  PARAMETER(ln_R0); 
  PARAMETER_VECTOR(ln_ycs_est);           // length(recruit_devs) = sum(ycs_estimated)
  PARAMETER(ln_sigma_r);                    // logistic fishery ogive
  PARAMETER( ln_extra_survey_cv );          // Additional survey cv.
  PARAMETER_VECTOR(trans_survey_error);     // 
  PARAMETER_VECTOR(trans_fishery_error);    // 
  
  PARAMETER(logit_f_a50);                   // logistic fishery ogive paramss
  PARAMETER(logit_f_ato95);                 // logistic fishery ogive paramss
  
  PARAMETER(logit_survey_a50);              // logistic survey ogive paramss
  PARAMETER(logit_survey_ato95);            // logistic survey ogive paramss
  // have trouble estiming these parameters with the Q so, need to bound them.
  PARAMETER(logit_surveyQ);                 // logit transformes bound between 0-1
  PARAMETER_VECTOR(ln_F);                   // length n_years
  PARAMETER(ln_catch_sd);
  /*
   * Parameter transformations
   */
  int year_ndx, age_ndx, iter, age_iter;
  Type extra_survey_cv = exp(ln_extra_survey_cv);
  Type R0 = exp(ln_R0);
  Type sigma_r = exp(ln_sigma_r);
  Type B0 = 0.0;
  Type catch_sd = exp(ln_catch_sd);
  array<Type> N(nages, nyears + 1);
  N.fill(0.0);
  // Convert mean length to mean weight
  array<Type> stockMeanWeight(stockMeanLength.dim[0], stockMeanLength.dim[1]);
  stockMeanWeight.fill(0.0);
  array<Type> catchMeanWeight(catchMeanLength.dim[0], catchMeanLength.dim[1]);
  stockMeanWeight.fill(0.0);
  
  for(iter = 0; iter < catchMeanLength.dim(0); ++iter) {
    for(age_iter = 0; age_iter < catchMeanLength.dim(1); ++age_iter) {
      stockMeanWeight(iter, age_iter) = mean_weight_a * pow(stockMeanLength(iter, age_iter), mean_weight_b);
      catchMeanWeight(iter, age_iter) = mean_weight_a * pow(catchMeanLength(iter, age_iter), mean_weight_b);
    }
  }
  
  // deal with YCS
  vector<Type> ycs(nyears);
  iter = 0;
  Type recruit_nll = 0.0;
  
  for(year_ndx = 0; year_ndx < nyears; ++year_ndx) {
    if (ycs_estimated[year_ndx] == 1) {
      ycs(year_ndx) = exp(ln_ycs_est(iter));
      ++iter;
    } else {
      ycs(year_ndx) = 1.0;
    }
  }
  if (standardise_ycs == 1) {
    ycs /= ycs.mean();
  } 
  // Note this contains constants (non estimated ycs values), and probably needs a jacombian for the transformation.
  // mean of random variables 
  for(year_ndx = 0; year_ndx < ln_ycs_est.size(); ++year_ndx) {
    //recruit_nll -= dnorm(lln_ycs_est(year_ndx), -0.5 * sigma_r * sigma_r, sigma_r, true) - ln_ycs_est(year_ndx);  // if random effect, will need this if Log-Normal distribution used
    //recruit_nll -= dnorm(ln_ycs_est(year_ndx), -0.5 * sigma_r * sigma_r, sigma_r, true);                          // if random effect, will need this if Log-Normal distribution used
    recruit_nll -= dnorm(ln_ycs_est(year_ndx), Type(0.0), sigma_r, true);                          // if random effect, will need this if Log-Normal distribution used
  }
  
  Type survey_a50 = invlogit_general(logit_survey_a50, sel_a50_bounds(0), sel_a50_bounds(1));
  Type survey_ato95 = invlogit_general(logit_survey_ato95, sel_ato95_bounds(0), sel_ato95_bounds(1));
  Type f_a50 = invlogit_general(logit_f_a50, sel_a50_bounds(0), sel_a50_bounds(1));
  Type f_ato95 = invlogit_general(logit_f_ato95, sel_ato95_bounds(0), sel_ato95_bounds(1));
  Type survey_Q = invlogit(logit_surveyQ);
  /*
   * Set up container storage
   */
  vector<Type> ssb(nyears + 1);
  ssb.setZero();
  vector<Type> annual_F = exp(ln_F);
  array<Type> F_ay(nages, nyears);
  F_ay.fill(0.0);
  array<Type> Z_ay(nages, nyears);
  Z_ay.fill(0.0);
  // Fitted value containers
  vector<Type> survey_index_fitted(survey_trans_comp.dim[1]);
  survey_index_fitted.fill(0.0);
  // If ALR comp, then need to adjust fitted value containers, because dims derived on input observation container
  int n_survey_ages = survey_trans_comp.dim[0];
  if((survey_comp_type == 0) & (survey_comp_likelihood == 0))
    n_survey_ages += 1;
  int n_fishery_ages = fishery_trans_comp.dim[0];
  if((fishery_comp_type == 0) & (fishery_comp_likelihood == 0))
    n_fishery_ages += 1;
  array<Type> survey_comp_fitted(n_survey_ages, survey_trans_comp.dim[1]);
  array<Type> survey_comp_trans_fitted(survey_trans_comp.dim[0], survey_trans_comp.dim[1]);
  matrix<Type> survey_trans_covar(survey_trans_comp.dim[0],survey_trans_comp.dim[0]);
  survey_trans_covar.setZero();
  get_covar(trans_survey_error, survey_trans_covar, survey_trans_comp.dim[0], survey_covar_structure);
  
  array<Type> fishery_comp_fitted(n_fishery_ages, fishery_trans_comp.dim[1]);
  array<Type> fishery_comp_trans_fitted(fishery_trans_comp.dim[0], fishery_trans_comp.dim[1]);
  matrix<Type> fishery_trans_covar(fishery_trans_comp.dim[0],fishery_trans_comp.dim[0]);
  fishery_trans_covar.setZero();
  get_covar(trans_fishery_error, fishery_trans_covar, fishery_trans_comp.dim[0], fishery_covar_structure);
  Type theta_fishery = exp(trans_fishery_error(0)); // dirichlet-multinomial parameter
  Type theta_survey = exp(trans_survey_error(0)); // dirichlet-multinomial parameter
  
  
  // ll densities
  MVNORM_t<Type> survey_mvnorm(survey_trans_covar); // Evaluate negative log likelihod
  MVNORM_t<Type> fishery_mvnorm(fishery_trans_covar); // Evaluate negative log likelihod
  
  vector<Type> predlogN(nages); 
  vector<Type> temp_partition(nages); 
  vector<Type> survey_partition(nages); 
  vector<Type> fishery_partition(nages); 
  
  vector<Type> pred_catches(nyears);
  pred_catches.setZero();
  vector<Type> survey_yearly_numbers(survey_trans_comp.dim[1]);
  survey_yearly_numbers.setZero();
  vector<Type> fishery_yearly_numbers(fishery_trans_comp.dim[1]);
  fishery_yearly_numbers.setZero();
  
  vector<Type> fishery_selectivity(nages);
  vector<Type> survey_selectivity(nages);
  vector<Type> survey_sd(survey_cv.size());
  
  for(iter = 0; iter < survey_sd.size(); ++iter) {
    survey_sd(iter) = sqrt(log(survey_cv(iter) * survey_cv(iter) + extra_survey_cv * extra_survey_cv + 1));
  }
  // Calculate vulnerable biomass and U
  Type survey_comp_nll = 0;
  Type survey_index_nll = 0;
  Type fishery_comp_nll = 0;
  Type catch_nll = 0.0;
  Type sum1 = 0.0;
  Type sum2 = 0.0;
  
  /*
   * Build Covariance's for Observations and states currently just iid with different covariances
   */
  
  /*
   * Deal with F stuff
   */
  for(age_ndx = 0; age_ndx < nages; ++age_ndx) {
    fishery_selectivity(age_ndx) = Type(1)/(Type(1) + pow(Type(19),((f_a50 - ages(age_ndx))/f_ato95)));
    survey_selectivity(age_ndx) = Type(1)/(Type(1) + pow(Type(19),((survey_a50 - ages(age_ndx))/survey_ato95)));
  }
  for(year_ndx = 0; year_ndx < nyears; year_ndx++) {
    F_ay.col(year_ndx) = fishery_selectivity * annual_F(year_ndx);
    Z_ay.col(year_ndx) = F_ay.col(year_ndx) + natMor;
  }
  /*
   * Initialise first year
   */
  // Initialise Partition
  for(age_ndx = 0; age_ndx < nages; ++age_ndx) 
    N(age_ndx, 0) = R0 * exp(-(ages(age_ndx)) * natMor);
  if(maxAgePlusGroup == 1)
    N(nages - 1, 0) = R0 * exp(- ages(nages - 1) * natMor) / (1.0 - exp(-natMor));
  // Applying ageing
  temp_partition = N.col(0);
  N((nages - 1), 0) += N(nages - 2, 0);
  for(age_ndx = 1; age_ndx < (nages - 1); ++age_ndx) 
    N(age_ndx, 0) =  temp_partition(age_ndx - 1, 0);
  N(0, 0) = R0;
  ssb(0) =  sum(N.col(0) * exp(-natMor * propZ_ssb(0))  * stockMeanWeight.col(0) * propMat.col(0));
  B0 = ssb(0);
  // Now the rest M
  //for(age_ndx = 0; age_ndx < nages; ++age_ndx) 
  //  N(age_ndx, 0) *= exp(-natMor);
  
  /*
   * Start Model
   */
  for(year_ndx = 1; year_ndx <= nyears; year_ndx++) {
    //----------------
    //Recuritment
    //----------------
    if(stockRecruitmentModelCode == 0) { // straight RW 
      predlogN(0) = log(N(0,year_ndx - 1) * ycs(year_ndx - 1));
    } else if(stockRecruitmentModelCode == 2) { //BH
      predlogN(0) = log(R0 * BevertonHolt(ssb(year_ndx - 1), B0, steepness) * ycs(year_ndx - 1));
    } else{
      error("SR model code not recognized");
    }
    
    //----------------
    // F + M + ageing
    //----------------
    for(age_ndx = 1; age_ndx < nages; ++age_ndx) 
      predlogN(age_ndx) = log(N(age_ndx - 1, year_ndx - 1)) - Z_ay(age_ndx - 1, year_ndx - 1);
    
    if(maxAgePlusGroup == 1) {
      predlogN(nages - 1) = log(N(nages - 2, year_ndx - 1) * exp(- Z_ay(nages - 2, year_ndx - 1)) +
        N(nages - 1, year_ndx - 1) * exp(-Z_ay(nages - 1, year_ndx - 1)));
    }  
    
    // Transform from log space
    N.col(year_ndx) = exp(predlogN);
    
    // Calculate SSBs an interpolation bewtween the year, starting with previous years Paritition
    for(age_ndx = 0; age_ndx < nages; ++age_ndx) {
      ssb(year_ndx) += N(age_ndx, year_ndx - 1) * exp(-Z_ay(age_ndx, year_ndx - 1) * propZ_ssb(year_ndx - 1)) * propMat(age_ndx, year_ndx - 1) * stockMeanWeight(age_ndx, year_ndx - 1);
      pred_catches(year_ndx - 1) += exp(log(N(age_ndx, year_ndx - 1)) + log(F_ay(age_ndx, year_ndx - 1)) - log(Z_ay(age_ndx, year_ndx - 1)) + log(1 - exp(-Z_ay(age_ndx, year_ndx - 1)))) * catchMeanWeight(age_ndx, year_ndx - 1);
    }
  }
  //
  // Observational Model
  // - Calculate Fitted values
  // - Calculate likelihods
  //
  // Numbers at age
  iter = 0;
  for(year_ndx = 0; year_ndx < nyears; year_ndx++) {
    if (survey_year_indicator(year_ndx) == 1) {
      survey_partition.fill(0.0);
      for(age_ndx = 0; age_ndx < nages; ++age_ndx) {
        survey_partition[age_ndx] = survey_Q * survey_selectivity(age_ndx) * N(age_ndx, year_ndx) * exp(-(Z_ay(age_ndx, year_ndx)) * propZ_survey(iter));
        survey_index_fitted(iter) += survey_partition[age_ndx] * stockMeanWeight(age_ndx, year_ndx);
      }
      // ageing error  
      temp_partition.fill(0.0);
      for (int a1 = 0; a1 < nages; ++a1) {
        for (int a2 = 0; a2 < nages; ++a2) 
          temp_partition[a2] += survey_partition(a1) * ageing_error_matrix(a1, a2);
      }
      // Truncate Obs
      age_iter = 0;
      for(age_ndx = 0; age_ndx < nages; ++age_ndx) {
        if((ages(age_ndx) >= survey_min_age) & (ages(age_ndx) <= survey_max_age)) {
          survey_comp_fitted(age_iter, iter) = temp_partition(age_ndx);
          survey_yearly_numbers(iter) += survey_comp_fitted(age_iter, iter);
          if(ages(age_ndx) < survey_max_age)
            ++age_iter;
        } else if ((ages(age_ndx) > survey_max_age) & (survey_age_plus_group == 1)) {
          survey_comp_fitted(age_iter, iter) += temp_partition(age_ndx);
          survey_yearly_numbers(iter) += temp_partition(age_ndx);
        }
      }
      ++iter;
    }
  }
  // Fishery 
  iter = 0;
  for(year_ndx = 0; year_ndx < nyears; year_ndx++) {
    if (fishery_year_indicator(year_ndx) == 1) {
      fishery_partition.fill(0.0);
      for(age_ndx = 0; age_ndx < nages; ++age_ndx)
        fishery_partition[age_ndx] = N(age_ndx, year_ndx)  * (1 - exp(- Z_ay(age_ndx, year_ndx))) * F_ay(age_ndx, year_ndx) / Z_ay(age_ndx, year_ndx);
      
      // ageing error  
      temp_partition.fill(0.0);
      for (int a1 = 0; a1 < nages; ++a1) {
        for (int a2 = 0; a2 < nages; ++a2) 
          temp_partition[a2] += fishery_partition(a1) * ageing_error_matrix(a1,a2);
      }
      age_iter = 0;
      for(age_ndx = 0; age_ndx < nages; ++age_ndx) {
        if((ages(age_ndx) >= fishery_min_age) & (ages(age_ndx) <= fishery_max_age)) {
          fishery_comp_fitted(age_iter, iter) = temp_partition(age_ndx);
          fishery_yearly_numbers(iter) += temp_partition(age_ndx);
          if(ages(age_ndx) < fishery_max_age)
            ++age_iter;
        } else if ((ages(age_ndx) > fishery_max_age) & (fishery_age_plus_group == 1)) {
          fishery_comp_fitted(age_iter, iter) += temp_partition(age_ndx);
          fishery_yearly_numbers(iter) += temp_partition(age_ndx);
        }
      }
      ++iter;
    }
  }
  
  // Evaluate ll and simulate if we want to
  iter = 0;
  for(year_ndx = 0; year_ndx < nyears; year_ndx++) {
    if (survey_year_indicator(year_ndx) == 1) {
      if (survey_comp_type == 1) {
        // Numbers at age log
        survey_comp_trans_fitted.col(iter) = log(survey_comp_fitted.col(iter));
        survey_comp_nll += survey_mvnorm(survey_trans_comp.col(iter) - survey_comp_trans_fitted.col(iter), keep_survey_comp.col(iter));
        SIMULATE {
          survey_trans_comp.col(iter) = survey_comp_trans_fitted.col(iter) + survey_mvnorm.simulate();
        }
      } else {
        // Prop at age + biomass index
        survey_index_nll -= survey_index_keep(iter) * dnorm(log(survey_obs(iter)), log(survey_index_fitted(iter)) - 0.5 * survey_sd(iter) * survey_sd(iter), survey_sd(iter), true);
        //std::cout << "iter = " << iter << " val = " << survey_index_nll << " lower = " << survey_index_keep.cdf_lower(iter) << " upper = " << survey_index_keep.cdf_upper(iter) << " pnorm = " << log( pnorm(log(survey_obs(iter)), log(survey_index_fitted(iter)) - 0.5 * survey_sd(iter) * survey_sd(iter), survey_sd(iter)) )<< "\n";
        /* // Starting conditions can really effect pnorm() calls causing NaN's 
         survey_index_nll -= survey_index_keep.cdf_lower(iter) * log( pnorm(log(survey_obs(iter)), log(survey_index_fitted(iter)) - 0.5 * survey_sd(iter) * survey_sd(iter), survey_sd(iter)) );
         //std::cout << "iter = " << iter << " val = " << survey_index_nll << "\n";
         survey_index_nll -= survey_index_keep.cdf_upper(iter) * log( 1.0 - pnorm(log(survey_obs(iter)), log(survey_index_fitted(iter)) - 0.5 * survey_sd(iter) * survey_sd(iter), survey_sd(iter)) );
         //std::cout << "iter = " << iter << " val = " << survey_index_nll << "\n";
         */
        SIMULATE {
          survey_obs(iter) = exp(rnorm(log(survey_index_fitted(iter)) - 0.5 * survey_sd(iter) * survey_sd(iter), survey_sd(iter)));
        }   
        survey_comp_fitted.col(iter) /= survey_yearly_numbers(iter);
        if (survey_comp_likelihood == 0) {
          // ALN transformation log( N_a / N_A)
          survey_comp_trans_fitted.col(iter) = log(survey_comp_fitted.col(iter).vec().head(survey_comp_trans_fitted.rows()) / survey_comp_fitted.col(iter).vec()(survey_comp_trans_fitted.rows()));
          // ll
          survey_comp_nll += survey_mvnorm(survey_trans_comp.col(iter) - survey_comp_trans_fitted.col(iter), keep_survey_comp.col(iter));
          // relative index simulate
          SIMULATE {
            survey_trans_comp.col(iter) = survey_comp_trans_fitted.col(iter) + survey_mvnorm.simulate();
          }
        } else if (survey_comp_likelihood == 1) {
          // Multinomial
          survey_comp_nll -= dmultinom(vector<Type>(survey_trans_comp.col(iter)), vector<Type>(survey_comp_fitted.col(iter)), true);
          SIMULATE {
            Type N_eff = sum(survey_trans_comp.col(iter));
            survey_trans_comp.col(iter) = rmultinom(survey_comp_fitted.col(iter).vec(), N_eff);
          }
        } else if (survey_comp_likelihood == 2) {
          //Multinomial-Dirichlet
          Type N_eff = sum(survey_trans_comp.col(iter));
          vector<Type> comp_prop = survey_trans_comp.col(iter).vec();
          comp_prop /= comp_prop.sum();
          sum1 = 0.0;
          sum2 = 0.0;
          for(int ndx = 0; ndx < comp_prop.size(); ndx++){
            sum1 += lgamma(N_eff * comp_prop(ndx) + 1);
            sum2 += lgamma(N_eff * comp_prop(ndx) + theta_survey * N_eff * survey_comp_fitted(ndx, iter)) - lgamma(theta_survey * N_eff * survey_comp_fitted(ndx, iter));
          }
          survey_comp_nll -= lgamma(N_eff + 1) - sum1 + lgamma(theta_survey * N_eff) - lgamma(N_eff + theta_survey * N_eff) + sum2;
          
          SIMULATE {
            survey_trans_comp.col(iter) = rdirichletmulti(survey_comp_fitted.col(iter).vec(), N_eff, theta_survey);
          }
        }
      }
      ++iter;
    }
  }
  
  iter = 0;
  for(year_ndx = 0; year_ndx < nyears; year_ndx++) {
    if (fishery_year_indicator(year_ndx) == 1) {
      if (fishery_comp_type == 1) {
        // Numbers at age log
        fishery_comp_trans_fitted.col(iter) = log(fishery_comp_fitted.col(iter));
        fishery_comp_nll += fishery_mvnorm(fishery_trans_comp.col(iter) - fishery_comp_trans_fitted.col(iter), keep_fishery_comp.col(iter));
        SIMULATE {
          fishery_trans_comp.col(iter) = fishery_comp_trans_fitted.col(iter) + fishery_mvnorm.simulate();
        }
      } else {
        fishery_comp_fitted.col(iter) /= fishery_yearly_numbers(iter);
        // Prop at age ALN
        if (fishery_comp_likelihood == 0) {
          fishery_comp_trans_fitted.col(iter) = log(fishery_comp_fitted.col(iter).vec().head(fishery_comp_trans_fitted.rows()) / fishery_comp_fitted.col(iter).vec()(fishery_comp_trans_fitted.rows()));
          fishery_comp_nll += fishery_mvnorm(fishery_trans_comp.col(iter) - fishery_comp_trans_fitted.col(iter), keep_fishery_comp.col(iter));
          SIMULATE {
            fishery_trans_comp.col(iter) = fishery_comp_trans_fitted.col(iter) + fishery_mvnorm.simulate();
          }
          // Prop at age Multinomial
        } else if (fishery_comp_likelihood == 1) {
          fishery_comp_nll -= dmultinom(vector<Type>(fishery_trans_comp.col(iter)), vector<Type>(fishery_comp_fitted.col(iter)), true);
          SIMULATE {
            Type N_eff = sum(fishery_trans_comp.col(iter));
            fishery_trans_comp.col(iter) = rmultinom(fishery_comp_fitted.col(iter).vec(), N_eff);
          }
        } else if (fishery_comp_likelihood == 2) {
          Type N_eff = sum(fishery_trans_comp.col(iter));
          vector<Type> comp_prop = fishery_trans_comp.col(iter).vec();
          comp_prop /= comp_prop.sum();
          sum1 = 0.0;
          sum2 = 0.0;
          for(int ndx = 0; ndx < comp_prop.size(); ndx++){
            sum1 += lgamma(N_eff * comp_prop(ndx) + 1);
            sum2 += lgamma(N_eff * comp_prop(ndx) + theta_fishery * N_eff * fishery_comp_fitted(ndx, iter)) - lgamma(theta_fishery * N_eff * fishery_comp_fitted(ndx, iter));
          }
          fishery_comp_nll -= lgamma(N_eff + 1) - sum1 + lgamma(theta_fishery * N_eff) - lgamma(N_eff + theta_fishery * N_eff) + sum2;
          SIMULATE {
            fishery_trans_comp.col(iter) = rdirichletmulti(fishery_comp_fitted.col(iter).vec(), N_eff, theta_fishery);
          }
        }
      }
      ++iter;
    }
  }
  
  catch_nll =  -1.0 * sum(dnorm(log(catches), log(pred_catches) - 0.5 * catch_sd * catch_sd, catch_sd, true));
  
  SIMULATE {
    vector<Type> log_mean_pred_catches = log(pred_catches) - 0.5 * catch_sd * catch_sd;
    catches = exp(rnorm(log_mean_pred_catches, catch_sd));
    REPORT(survey_trans_comp);
    REPORT(fishery_trans_comp);
    REPORT(catches);
    if (survey_comp_type == 0) 
      REPORT( survey_obs );
  }
  
  
  Type joint_nll = fishery_comp_nll + survey_comp_nll + survey_index_nll + recruit_nll + catch_nll;
  if (isNA(joint_nll))
    error("joint_nll = NA");
  //std::cout << "Joint ll = " << joint_ll << " catch pen1 = " << catch_pen << " catch pen2 = " << catch_pen2 <<"\n";
  REPORT( catch_nll );
  REPORT( ssb );
  REPORT( R0 );
  REPORT( sigma_r );
  REPORT( B0 );
  REPORT( fishery_comp_nll );
  REPORT( survey_comp_nll );
  REPORT( joint_nll );
  REPORT( recruit_nll );
  //REPORT( annual_Fs );
  REPORT( pred_catches );
  
  REPORT( fishery_selectivity );
  REPORT( survey_selectivity );
  
  REPORT( N );
  REPORT( ycs );
  
  REPORT( survey_a50 );
  REPORT( survey_ato95 ); 
  REPORT( survey_Q );
  REPORT( f_a50 );
  REPORT( f_ato95 );
  REPORT( annual_F );
  REPORT( extra_survey_cv );
  REPORT( survey_sd );
  REPORT( survey_trans_covar );
  REPORT( survey_comp_fitted );
  REPORT( survey_comp_trans_fitted );
  if (survey_comp_type == 0) {
    REPORT( survey_index_fitted );
    REPORT( survey_index_nll );
  }
  REPORT( catch_sd );
  REPORT( fishery_trans_covar );
  REPORT( fishery_comp_fitted );
  REPORT( fishery_comp_trans_fitted );
  
  REPORT( F_ay );  
  REPORT( Z_ay );
  ADREPORT( ssb );
  
  
  REPORT(stockMeanWeight);
  REPORT(catchMeanWeight);
  
  REPORT( theta_fishery );
  REPORT( theta_survey );
  
  // ADREPORT(logN);
  // ADREPORT(logF);
  return joint_nll; // we are minimising with nlminb
  //return 0.0;
}
