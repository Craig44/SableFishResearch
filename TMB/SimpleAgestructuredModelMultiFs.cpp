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
// transform Y -Inf-Inf -> X bound lb - ub
template <class Type> 
vector<Type> invlogit_general(vector<Type>& Y, Type& lb, Type& ub) {
  vector<Type> X(Y.size());
  for(int i = 0; i < X.size(); ++i) {
    X(i) = lb + (ub - lb) * (1 / (1 + exp(-Y(i))));
  }
  return(X);
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
  int n_years = years.size();
  int n_ages = ages.size();
  DATA_INTEGER(maxAgePlusGroup);  // 1 = yes, 0 = no
  DATA_INTEGER(n_fisheries);      // number of fisheries
  // Observation info
  DATA_MATRIX(ageing_error_matrix);     // n_ages * n_ages
  DATA_IVECTOR(survey_year_indicator);  // 1 = calculate, 0 = ignore n_years
  DATA_VECTOR(survey_obs);              // Relative index (this will be ignored if survey_comp_type = 1)
  DATA_VECTOR(survey_cv);               // CV for relative index (this will be ignored if survey_comp_type = 1)
  DATA_VECTOR_INDICATOR(survey_index_keep, survey_obs);  // For one-step predictions
  
  DATA_ARRAY(survey_AF_obs);            // numbers at age. dim: n_ages x sum(survey_year_indicator) 
  DATA_ARRAY_INDICATOR(keep_survey_comp, survey_AF_obs);
  
  DATA_IARRAY(fishery_year_indicator);  // 1 = calculate, 0 = ignore,   nyear x nfisheries
  DATA_ARRAY(fishery_AF_obs);        // numbers at age. dim: n_ages x sum(fishery_year_indicator) * n_fisheries
  DATA_ARRAY_INDICATOR(keep_fishery_comp, fishery_AF_obs);
  
  DATA_IVECTOR(ycs_estimated);    // 1 = estimated, 0 = ignore
  DATA_INTEGER(standardise_ycs);  // 1 = yes, 0 = No
  
  // Catch info
  DATA_IARRAY(catch_indicator);       // length = n_years x n_fisheries. 0 = no catch this year for fishery, 1 = yes catch
  DATA_ARRAY(catches);                // length = n_years x n_fisheries
  DATA_SCALAR(F_max);                 // max F = 2.56
  DATA_INTEGER(F_iterations);         // should be between 2-5
  DATA_INTEGER(F_method );            // 0 = estimate F's as free parameters, 1 = Hybrid method
  
  DATA_VECTOR(propZ_ssb);             // proportion of Z for SSB, length n_years
  DATA_VECTOR(propZ_survey);          // proportion of Z for SSB, length n_years
  // Biological info
  DATA_ARRAY(propMat);                // Proportion Mature (used in SSB)    dim = n_ages x n_years
  DATA_ARRAY(stockMeanLength);        // Stock Mean weight used in SSB + survey calculation  dim = n_ages x n_years
  DATA_ARRAY(catchMeanLength);        // Stock Mean weight used in Catch equation calculation dim = n_ages x n_years
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
  
  PARAMETER_VECTOR(logit_f_a50);            // logistic fishery ogive parameters for each fishery
  PARAMETER_VECTOR(logit_f_ato95);          // logistic fishery ogive parameters for each fishery
  
  PARAMETER(logit_survey_a50);              // logistic survey ogive paramss
  PARAMETER(logit_survey_ato95);            // logistic survey ogive paramss
  // have trouble estiming these parameters with the Q so, need to bound them.
  PARAMETER(logit_surveyQ);                 // logit transformes bound between 0-1
  PARAMETER_ARRAY(ln_F);                    // length n_years x n_fisheries
  PARAMETER(ln_catch_sd);
  /*
   * Parameter transformations
   */
  int year_ndx, age_ndx, iter, fishery_ndx;
  Type extra_survey_cv = exp(ln_extra_survey_cv);
  Type R0 = exp(ln_R0);
  Type sigma_r = exp(ln_sigma_r);
  Type B0 = 0.0;
  Type catch_sd = exp(ln_catch_sd);
  array<Type> N(n_ages, n_years + 1);
  N.fill(0.0);
  // Convert mean length to mean weight
  array<Type> stockMeanWeight(stockMeanLength.dim);
  stockMeanWeight.fill(0.0);
  array<Type> catchMeanWeight(catchMeanLength.dim);
  stockMeanWeight.fill(0.0);
  
  for(iter = 0; iter < catchMeanLength.dim(0); ++iter) {
    for(age_ndx = 0; age_ndx < catchMeanLength.dim(1); ++age_ndx) {
      stockMeanWeight(iter, age_ndx) = mean_weight_a * pow(stockMeanLength(iter, age_ndx), mean_weight_b);
      catchMeanWeight(iter, age_ndx) = mean_weight_a * pow(catchMeanLength(iter, age_ndx), mean_weight_b);
    }
  }
  
  // deal with YCS
  vector<Type> ycs(n_years);
  iter = 0;
  Type recruit_nll = 0.0;
  
  for(year_ndx = 0; year_ndx < n_years; ++year_ndx) {
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
  vector<Type> f_a50 = invlogit_general(logit_f_a50, sel_a50_bounds(0), sel_a50_bounds(1));
  vector<Type> f_ato95 = invlogit_general(logit_f_ato95, sel_ato95_bounds(0), sel_ato95_bounds(1));
  Type survey_Q = invlogit(logit_surveyQ);
  
  /*
   * Set up container storage
   */
  
  vector<Type> ssb(n_years + 1);
  ssb.setZero();
  array<Type> annual_Fs(n_fisheries, n_years);

  if(F_method == 0)
    annual_Fs = exp(ln_F);
  // else with the hybrid method we need to calculate it on the fly
  array<Type> F_ayf(n_ages, n_years, n_fisheries);
  F_ayf.fill(0.0);
  array<Type> Z_ay(n_ages, n_years);
  Z_ay.fill(natMor);
  // Fitted value containers
  vector<Type> survey_index_fitted(survey_obs.size());
  survey_index_fitted.fill(0.0);
  // If ALR comp, then need to adjust fitted value containers, because dims derived on input observation container
  array<Type> survey_AF_fitted(survey_AF_obs.dim);
  
  array<Type> fishery_AF_fitted(fishery_AF_obs.dim);
  
  
  // ll densities
  vector<Type> predlogN(n_ages); 
  vector<Type> current_partition(n_ages); 
  vector<Type> temp_partition(n_ages); 
  vector<Type> temp_partition_obs(n_ages); 
  
  vector<Type> survey_partition(n_ages); 
  vector<Type> fishery_partition(n_ages); 
  
  array<Type> pred_catches(n_years, n_fisheries);
  pred_catches.fill(0.0);
  
  vector<Type> survey_yearly_numbers(survey_AF_obs.dim[1]);
  survey_yearly_numbers.setZero();
  array<Type> fishery_yearly_numbers(fishery_AF_obs.dim[1], fishery_AF_obs.dim[2]);
  
  array<Type> fishery_selectivity(n_ages, n_fisheries);
  
  vector<Type> survey_selectivity(n_ages);
  vector<Type> survey_sd(survey_cv.size());
  
  for(iter = 0; iter < survey_sd.size(); ++iter) {
    survey_sd(iter) = sqrt(log(survey_cv(iter) * survey_cv(iter) + extra_survey_cv * extra_survey_cv + 1));
  }
  // Calculate vulnerable biomass and U
  Type survey_comp_nll = 0;
  Type survey_index_nll = 0;
  Type fishery_comp_nll = 0;
  Type catch_nll = 0.0;
  
  /*
   * Build Covariance's for Observations and states currently just iid with different covariances
   */
  
  /*
   * Deal with F stuff
   */
  
  for(age_ndx = 0; age_ndx < n_ages; ++age_ndx)
    survey_selectivity(age_ndx) = Type(1)/(Type(1) + pow(Type(19),((survey_a50 - ages(age_ndx))/survey_ato95)));
  
  for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx) {
    for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) {
      fishery_selectivity(age_ndx, fishery_ndx) = Type(1)/(Type(1) + pow(Type(19),((f_a50(fishery_ndx) - ages(age_ndx))/f_ato95(fishery_ndx))));
    }
  }
  // pre cache these values as we are estimating them and so have them available
  if(F_method == 0) {
    for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx) {
      for(year_ndx = 0; year_ndx < n_years; year_ndx++) {
        for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) {
          F_ayf(age_ndx, year_ndx, fishery_ndx) = fishery_selectivity(age_ndx) * annual_Fs(fishery_ndx, year_ndx);
          Z_ay(age_ndx, year_ndx) += F_ayf(age_ndx, year_ndx, fishery_ndx);
        }
      }
    }
  }
  /*
   * Initialise first year
   */
  // Initialise Partition
  for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) 
    N(age_ndx, 0) = R0 * exp(-(ages(age_ndx)) * natMor);
  if(maxAgePlusGroup == 1)
    N(n_ages - 1, 0) = R0 * exp(- ages(n_ages - 1) * natMor) / (1.0 - exp(-natMor));
  // Applying ageing
  temp_partition = N.col(0);
  N((n_ages - 1), 0) += N(n_ages - 2, 0);
  for(age_ndx = 1; age_ndx < (n_ages - 1); ++age_ndx) 
    N(age_ndx, 0) =  temp_partition(age_ndx - 1, 0);
  N(0, 0) = R0;
  ssb(0) =  sum(N.col(0) * exp(-natMor * propZ_ssb(0))  * stockMeanWeight.col(0) * propMat.col(0));
  B0 = ssb(0);
  // Now the rest M
  //for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) 
  //  N(age_ndx, 0) *= exp(-natMor);
  
  
  vector<Type> vulnerable_bio(n_fisheries);
  vector<Type> init_popes_rate(n_fisheries);
  vector<Type> steep_jointer(n_fisheries);
  vector<Type> init_F(n_fisheries);
  vector<Type> temp_Z_vals(n_ages);
  vector<Type> survivorship(n_ages);
  
  vector<Type> exploitation_rate(n_fisheries);
  Type exp_half_m =  exp(-natMor*0.5); // called many times in hybrid F calculation, good to cache
  Type interim_total_catch = 0;// interim catch over all fisheries in current year
  Type total_catch_this_year = 0;
  Type z_adjustment;
  /*
   * Start Model
   */
  
  for(year_ndx = 1; year_ndx <= n_years; year_ndx++) {
    vulnerable_bio.setZero();
    temp_partition = N.col(year_ndx - 1);
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
    //temp_partition(0) = exp(predlogN(0));
    // hybrid F algorithm
    // - calculate vulnerable/initF
    // - For f_iterations
    // -- calculate Z and lambda (survivorship)
    // -- adjust
    if(F_method == 1) {
      // calculate vulnerable and initial calculations
      for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx) {
        if(fishery_ndx == 0) {
          total_catch_this_year = catches(year_ndx - 1, fishery_ndx);
        } else {
          total_catch_this_year += catches(year_ndx - 1, fishery_ndx);
        }
        vulnerable_bio(fishery_ndx) = (N.col(year_ndx - 1).vec() * fishery_selectivity.col(fishery_ndx).vec() * exp_half_m * catchMeanWeight.col(year_ndx - 1).vec()).sum();
        init_popes_rate(fishery_ndx) = catches(year_ndx - 1, fishery_ndx) / (vulnerable_bio(fishery_ndx) + 0.1 * catches(year_ndx - 1, fishery_ndx)); //  Pope's rate  robust A.1.22 of SS appendix
        steep_jointer(fishery_ndx) = 1.0 / (1.0 + exp(30.0 * (init_popes_rate(fishery_ndx) - 0.95))); // steep logistic joiner at harvest rate of 0.95 //steep logistic joiner at harvest rate of 0.95
        exploitation_rate(fishery_ndx) = steep_jointer(fishery_ndx)  * init_popes_rate(fishery_ndx) + (1.0 - steep_jointer(fishery_ndx) ) * 0.95;
        init_F(fishery_ndx) = -log(1.0 - exploitation_rate(fishery_ndx));
      }
      
      // Now solve;
      for(int f_iter = 0; f_iter < F_iterations; ++f_iter) {
        
        interim_total_catch = 0;
        //initialise Z container with M
        for(age_ndx = 0; age_ndx < n_ages; ++age_ndx)
          temp_Z_vals(age_ndx) = natMor;
        // Use calculate an initial Z
        for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx) {
          for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) {
            temp_Z_vals(age_ndx) += init_F(fishery_ndx) * fishery_selectivity(age_ndx, fishery_ndx);
          }
        }
        // The survivorship is calculated as:
        for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) {
          survivorship(age_ndx) = (1.0 - exp(-temp_Z_vals(age_ndx))) / temp_Z_vals(age_ndx);
        }
        
        // Calculate the expected total catch that would occur with the current Hrates and Z
        for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx)  {
          interim_total_catch += (N.col(year_ndx - 1).vec() * init_F(fishery_ndx) * fishery_selectivity.col(fishery_ndx).vec() * survivorship * catchMeanWeight.col(year_ndx - 1).vec()).sum();
        }
        // make Z adjustments
        z_adjustment = total_catch_this_year / (interim_total_catch + 0.0001);
        //interim_catch_store(year_ndx - 1, f_iter, fishery_ndx) = interim_total_catch;
        for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) {
          temp_Z_vals(age_ndx) = natMor + z_adjustment * (temp_Z_vals(age_ndx) - natMor);
          survivorship(age_ndx) = (1.0 - exp(-temp_Z_vals(age_ndx))) / temp_Z_vals(age_ndx);
        }
        
        // Now re-calculate a new pope rate using a vulnerable biomass based
        // on the newly adjusted F
        for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx) {
          vulnerable_bio(fishery_ndx) = (N.col(year_ndx - 1).vec() * fishery_selectivity.col(fishery_ndx).vec() * catchMeanWeight.col(year_ndx - 1).vec() * survivorship).sum() ;
          exploitation_rate(fishery_ndx) = catches(year_ndx - 1, fishery_ndx) / (vulnerable_bio(fishery_ndx) + 0.0001); //  Pope's rate  robust A.1.22 of SS appendix
          steep_jointer(fishery_ndx) = 1.0 / (1.0 + exp(30.0 * (exploitation_rate(fishery_ndx) - 0.95 * F_max))); // steep logistic joiner at harvest rate of 0.95 //steep logistic joiner at harvest rate of 0.95
          init_F(fishery_ndx) = steep_jointer(fishery_ndx) * exploitation_rate(fishery_ndx) + (1.0 - steep_jointer(fishery_ndx)) * F_max; 
          annual_Fs(fishery_ndx, year_ndx - 1) = init_F(fishery_ndx);
        }
      }
       
      // Cache F and Z calculations
      for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx) {
       for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) {
         F_ayf(age_ndx, year_ndx - 1, fishery_ndx) = fishery_selectivity(age_ndx, fishery_ndx) * annual_Fs(fishery_ndx, year_ndx - 1);
         Z_ay(age_ndx, year_ndx - 1) +=  F_ayf(age_ndx, year_ndx - 1, fishery_ndx);
        }
      }
    }
    
    // Calculate SSBs an interpolation bewtween the year, starting with previous years Paritition
    for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) {
      ssb(year_ndx) += N(age_ndx, year_ndx - 1) * exp(-Z_ay(age_ndx, year_ndx - 1) * propZ_ssb(year_ndx - 1)) * propMat(age_ndx, year_ndx - 1) * stockMeanWeight(age_ndx, year_ndx - 1);
      for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx)
        pred_catches(year_ndx - 1, fishery_ndx) +=  F_ayf(age_ndx, year_ndx - 1, fishery_ndx) / Z_ay(age_ndx, year_ndx - 1) * N(age_ndx, year_ndx - 1) * (1 - exp(-Z_ay(age_ndx, year_ndx - 1))) * catchMeanWeight(age_ndx, year_ndx - 1);
    }
    //----------------
    // Apply F + M + ageing
    //----------------
    for(age_ndx = 1; age_ndx < n_ages; ++age_ndx) 
      predlogN(age_ndx) = log(N(age_ndx - 1, year_ndx - 1)) - Z_ay(age_ndx - 1, year_ndx - 1);
    
    if(maxAgePlusGroup == 1) {
      predlogN(n_ages - 1) = log(N(n_ages - 2, year_ndx - 1) * exp(- Z_ay(n_ages - 2, year_ndx - 1)) +
        N(n_ages - 1, year_ndx - 1) * exp(-Z_ay(n_ages - 1, year_ndx - 1)));
    }  
    
    // Transform from log space
    N.col(year_ndx) = exp(predlogN);

  }
  //
  // Observational Model
  // - Calculate Fitted values
  // - Calculate likelihods
  //
  // Numbers at age
  
  iter = 0;
  for(year_ndx = 0; year_ndx < n_years; year_ndx++) {
    if (survey_year_indicator(year_ndx) == 1) {
      survey_partition.fill(0.0);
      for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) {
        survey_partition[age_ndx] = survey_Q * survey_selectivity(age_ndx) * N(age_ndx, year_ndx) * exp(-(Z_ay(age_ndx, year_ndx)) * propZ_survey(iter));
        survey_index_fitted(iter) += survey_partition[age_ndx] * stockMeanWeight(age_ndx, year_ndx);
      }
      // ageing error  
      temp_partition.fill(0.0);
      for (int a1 = 0; a1 < n_ages; ++a1) {
        for (int a2 = 0; a2 < n_ages; ++a2) 
          temp_partition[a2] += survey_partition(a1) * ageing_error_matrix(a1, a2);
      }
      survey_AF_fitted.col(iter) += temp_partition;
      survey_yearly_numbers(iter) = sum(temp_partition);
      ++iter;
    }
  }
  // Fishery 
  
  for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx) {
    iter = 0;
    for(year_ndx = 0; year_ndx < n_years; year_ndx++) {
      if (fishery_year_indicator(year_ndx, fishery_ndx) == 1) {
        fishery_partition.fill(0.0);
        for(age_ndx = 0; age_ndx < n_ages; ++age_ndx)
          fishery_partition[age_ndx] = N(age_ndx, year_ndx)  * (1 - exp(- Z_ay(age_ndx, year_ndx))) * F_ayf(age_ndx, year_ndx, fishery_ndx) / Z_ay(age_ndx, year_ndx);
        // ageing error  
        temp_partition.fill(0.0);
        for (int a1 = 0; a1 < n_ages; ++a1) {
          for (int a2 = 0; a2 < n_ages; ++a2) 
            temp_partition[a2] += fishery_partition(a1) * ageing_error_matrix(a1,a2);
        }
        for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) {
          fishery_AF_fitted(age_ndx, iter, fishery_ndx) = temp_partition(age_ndx);
        }
        fishery_yearly_numbers(iter, fishery_ndx) = sum(temp_partition);
      }
      ++iter; 
    }
  }
  
  // Evaluate ll and simulate if we want to
  iter = 0;
  for(year_ndx = 0; year_ndx < n_years; year_ndx++) {
    if (survey_year_indicator(year_ndx) == 1) {
      // Prop at age + biomass index
      survey_index_nll -= survey_index_keep(iter) * dnorm(log(survey_obs(iter)), log(survey_index_fitted(iter)) - 0.5 * survey_sd(iter) * survey_sd(iter), survey_sd(iter), true);
      //std::cout << "iter = " << iter << " val = " << survey_index_nll << " lower = " << survey_index_keep.cdf_lower(iter) << " upper = " << survey_index_keep.cdf_upper(iter) << " pnorm = " << log( pnorm(log(survey_obs(iter)), log(survey_index_fitted(iter)) - 0.5 * survey_sd(iter) * survey_sd(iter), survey_sd(iter)) )<< "\n";
      SIMULATE {
        survey_obs(iter) = exp(rnorm(log(survey_index_fitted(iter)) - 0.5 * survey_sd(iter) * survey_sd(iter), survey_sd(iter)));
      }   
      survey_AF_fitted.col(iter) /= survey_yearly_numbers(iter);
      // Multinomial
      survey_comp_nll -= dmultinom(vector<Type>(survey_AF_obs.col(iter)), vector<Type>(survey_AF_fitted.col(iter)), true);
      SIMULATE {
        Type N_eff = sum(survey_AF_obs.col(iter));
        survey_AF_obs.col(iter) = rmultinom(survey_AF_fitted.col(iter).vec(), N_eff);
      }
      ++iter;
    }
  }
  
  
  
  for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx) {
    iter = 0;
    for(year_ndx = 0; year_ndx < n_years; year_ndx++) {
      if (fishery_year_indicator(year_ndx, fishery_ndx) == 1) {
        for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) {
          fishery_AF_fitted(age_ndx, iter, fishery_ndx) /= fishery_yearly_numbers(iter, fishery_ndx);
          temp_partition(age_ndx) = fishery_AF_fitted(age_ndx, iter, fishery_ndx);
          temp_partition_obs(age_ndx) = fishery_AF_obs(age_ndx, iter, fishery_ndx);
        }
        fishery_comp_nll -= dmultinom(temp_partition_obs, temp_partition, true);
        SIMULATE {
          Type N_eff = sum(temp_partition_obs);
          temp_partition_obs = rmultinom(temp_partition, N_eff);
          for(age_ndx = 0; age_ndx < n_ages; ++age_ndx) 
            fishery_AF_obs(age_ndx, iter, fishery_ndx) = temp_partition_obs(age_ndx);
        }
        ++iter;
      }
    }
  }
  for(year_ndx = 0; year_ndx < n_years; year_ndx++) {
    for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx) 
      catch_nll -= dnorm(log(catches(year_ndx, fishery_ndx)), log(pred_catches(year_ndx, fishery_ndx)) - 0.5 * catch_sd * catch_sd, catch_sd, true);
  }
  SIMULATE {
    for(year_ndx = 0; year_ndx < n_years; year_ndx++) {
      for(fishery_ndx = 0; fishery_ndx < n_fisheries; ++fishery_ndx) 
        catches(year_ndx, fishery_ndx) = exp(rnorm(log(pred_catches(year_ndx, fishery_ndx)) - 0.5 * catch_sd * catch_sd, catch_sd));
    }
    REPORT(survey_AF_obs);
    REPORT(fishery_AF_obs);
    REPORT(catches);
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
  //REPORT( annual_Fss );
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
  REPORT( annual_Fs );
  REPORT( extra_survey_cv );
  REPORT( survey_sd );
  REPORT( survey_AF_fitted );
  REPORT( survey_index_fitted );
  REPORT( survey_index_nll );
  
  REPORT( catch_sd );
  REPORT( fishery_AF_fitted );
  
  REPORT( F_ayf );  
  REPORT( Z_ay );
  ADREPORT( ssb );
  
  
  REPORT(stockMeanWeight);
  REPORT(catchMeanWeight);
  
  // ADREPORT(logN);
  // ADREPORT(logF);
  return joint_nll; // we are minimising with nlminb
  //return 0.0;
}
