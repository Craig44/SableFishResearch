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
  int nyears = years.size();
  int nages = ages.size();
  DATA_INTEGER(maxAgePlusGroup);  // 1 = yes, 0 = no
  
  // Observation info
  DATA_MATRIX(ageing_error_matrix);     // nages * nages
  DATA_IVECTOR(survey_year_indicator);  // 1 = calculate, 0 = ignore nyears
  DATA_VECTOR(survey_obs);              // Relative index (this will be ignored if survey_comp_type = 1)
  DATA_VECTOR(survey_cv);               // CV for relative index (this will be ignored if survey_comp_type = 1)
  DATA_VECTOR_INDICATOR(survey_index_keep, survey_obs);  // For one-step predictions
  
  DATA_ARRAY(survey_AF_obs);            // numbers of fish dimensions= n_ages x 2 (males then females) : n_survey_years
  DATA_INTEGER(survey_AF_type);         // 0 = sex disaggregated (each sex sums = 1), 1 = 
  DATA_VECTOR(survey_numbers_male);  // if  survey_AF_type = 0, then this observation is used in the joint negative loglikelihood 
  
  DATA_IVECTOR(fishery_year_indicator);  // 1 = calculate, 0 = ignore
  DATA_ARRAY(fishery_AF_obs);            // numbers of fish dimensions= n_ages x 2 (males then females) : n_survey_years
  DATA_INTEGER(fishery_AF_type);         // 0 = sex disaggregated (each sex sums = 1), 1 = 
  DATA_VECTOR(fishery_numbers_male);  // if  fishery_AF_type = 0, then this observation is used in the joint negative loglikelihood 
  
  
  DATA_IVECTOR(ycs_estimated);    // 1 = estimated, 0 = ignore
  DATA_INTEGER(standardise_ycs);  // 1 = yes, 0 = No
  
  // Catch info
  DATA_VECTOR(catches);               // length = nyears
  DATA_VECTOR(propZ_ssb);             // proportion of Z for SSB, length nyears
  DATA_VECTOR(propZ_survey);          // proportion of Z for SSB, length nyears
  // Biological info
  DATA_ARRAY(prop_mature);     // Proportion of Female mature dim = nages x nyears
  DATA_ARRAY(stockMeanLength);        // Stock Mean weight used in SSB + survey calculation  dim =  nages x nyears x n_sex
  DATA_ARRAY(catchMeanLength);        // Stock Mean weight used in Catch equation calculation dim = nages x nyears x n_sex
  DATA_SCALAR(natMor);                // Instantaneous natural mortality        
  DATA_SCALAR(steepness);             // Instantaneous natural mortality     
  DATA_VECTOR(mean_weight_a);         // a parameter mean weight from length male then female
  DATA_VECTOR(mean_weight_b);         // b parameter mean weight from length male then female    

  DATA_INTEGER(stockRecruitmentModelCode); // SR relationship 0 = RW, 1 = Ricker, 2 = BH
  
  // Bounds for selectivity parameters for model stability
  DATA_VECTOR(sel_ato95_bounds);  // length 2
  DATA_VECTOR(sel_a50_bounds);    // length 2
  DATA_VECTOR(sel_alpha_bounds);    // length 2
  
  
  /*
   * Parameters to estiamte.
   */
  PARAMETER(ln_R0); 
  PARAMETER_VECTOR(ln_ycs_est);              // length(recruit_devs) = sum(ycs_estimated)
  PARAMETER(ln_sigma_r);                     // logistic fishery ogive
  PARAMETER( ln_extra_survey_cv );           // Additional survey cv.

  PARAMETER_VECTOR(logit_f_a50);                   // logistic fishery ogive paramss
  PARAMETER_VECTOR(logit_f_ato95);                 // logistic fishery ogive paramss
  PARAMETER(logit_f_alpha_f);                 // is female more or less selective compared with males
  
  PARAMETER_VECTOR(logit_survey_a50);              // logistic survey ogive paramss
  PARAMETER_VECTOR(logit_survey_ato95);            // logistic survey ogive paramss
  PARAMETER(logit_survey_alpha_f);            // is female more or less selective compared with males
  // have trouble estiming these parameters with the Q so, need to bound them.
  PARAMETER(logit_surveyQ);                 // logit transformes bound between 0-1
  PARAMETER_VECTOR(ln_F);                   // length n_years
  PARAMETER(ln_catch_sd);
  PARAMETER_VECTOR(logit_proportion_male);   // length(recruit_devs) = sum(ycs_estimated)
  
  /*
   * Parameter transformations
   */
  int year_ndx, age_ndx, iter, sex_iter;
  Type extra_survey_cv = exp(ln_extra_survey_cv);
  Type R0 = exp(ln_R0);
  Type sigma_r = exp(ln_sigma_r);
  Type B0 = 0.0;
  Type catch_sd = exp(ln_catch_sd);
  array<Type> N(nages, nyears + 1, 2);
  N.fill(0.0);
  // Convert mean length to mean weight
  array<Type> stockMeanWeight(stockMeanLength.dim);
  stockMeanWeight.fill(0.0);
  array<Type> catchMeanWeight(catchMeanLength.dim);
  catchMeanWeight.fill(0.0);
  
  
  for(age_ndx = 0; age_ndx < catchMeanLength.dim(0); ++age_ndx) {
    for(year_ndx = 0; year_ndx < catchMeanLength.dim(1); ++year_ndx) {
      for(sex_iter = 0; sex_iter < 2; ++sex_iter) {
        stockMeanWeight(age_ndx, year_ndx, sex_iter) = mean_weight_a(sex_iter) * pow(stockMeanLength(age_ndx, year_ndx, sex_iter), mean_weight_b(sex_iter));
        catchMeanWeight(age_ndx, year_ndx, sex_iter) = mean_weight_a(sex_iter) * pow(catchMeanLength(age_ndx, year_ndx, sex_iter), mean_weight_b(sex_iter));
      }
    }
  }
  
  // deal with YCS
  vector<Type> ycs(nyears);
  iter = 0;
  Type recruit_nll = 0.0;
  
  for(year_ndx = 0; year_ndx < nyears; ++year_ndx) {
    if (ycs_estimated[year_ndx] == 1) {
      ycs(year_ndx) = exp(ln_ycs_est(iter)) + 0.5 * sigma_r * sigma_r;
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
  
  /*
   * Set up container storage
   */
  vector<Type> survey_a50 = invlogit_general(logit_survey_a50, sel_a50_bounds(0), sel_a50_bounds(1));
  vector<Type> survey_ato95 = invlogit_general(logit_survey_ato95, sel_ato95_bounds(0), sel_ato95_bounds(1));
  vector<Type> f_a50 = invlogit_general(logit_f_a50, sel_a50_bounds(0), sel_a50_bounds(1));
  vector<Type> f_ato95 = invlogit_general(logit_f_ato95, sel_ato95_bounds(0), sel_ato95_bounds(1));
  Type survey_alpha_f = invlogit_general(logit_survey_alpha_f, sel_alpha_bounds(0), sel_alpha_bounds(1));
  Type f_alpha_f = invlogit_general(logit_f_alpha_f, sel_alpha_bounds(0), sel_alpha_bounds(1));
  
  Type survey_Q = invlogit(logit_surveyQ);
  
  vector<Type> proportion_male = invlogit(logit_proportion_male);
  
  vector<Type> ssb(nyears + 1);
  ssb.setZero();
  vector<Type> annual_F = exp(ln_F);
  array<Type> F_ay(nages, nyears, 2);
  F_ay.fill(0.0);
  array<Type> Z_ay(nages, nyears, 2);
  Z_ay.fill(0.0);
  // Fitted value containers
  vector<Type> survey_index_fitted(sum(survey_year_indicator));
  survey_index_fitted.fill(0.0);
  // If ALR comp, then need to adjust fitted value containers, because dims derived on input observation container

  array<Type> survey_comp_fitted(survey_AF_obs.dim);
  array<Type> fishery_comp_fitted(fishery_AF_obs.dim);
  array<Type> survey_numbers_by_sex(survey_numbers_male.size(), 2); 
  array<Type> fishery_numbers_by_sex(fishery_numbers_male.size(), 2); 
  survey_numbers_by_sex.fill(0.0);
  fishery_numbers_by_sex.fill(0.0);
  vector<Type> survey_proportion_male_fitted(survey_numbers_male.size()); 
  vector<Type> fishery_proportion_male_fitted(fishery_numbers_male.size()); 
  
  
  
  array<Type> predlogN(nages,2); 
  vector<Type> temp_partition_m(nages); 
  vector<Type> temp_partition_f(nages); 
  vector<Type> temp_partition(nages); 
  array<Type> survey_partition(nages, 2); 
   array<Type> fishery_partition(nages, 2); 
  
  vector<Type> pred_catches(nyears);
  vector<Type> annual_recruitment(nyears);
  pred_catches.setZero();
  vector<Type> survey_yearly_numbers(survey_AF_obs.dim[1]);
  survey_yearly_numbers.setZero();
  vector<Type> fishery_yearly_numbers(fishery_AF_obs.dim[1]);
  fishery_yearly_numbers.setZero();
  
  array<Type> fishery_selectivity(nages, 2);
  array<Type> survey_selectivity(nages, 2);
  vector<Type> survey_sd(survey_cv.size());
  
  for(iter = 0; iter < survey_sd.size(); ++iter) {
    survey_sd(iter) = sqrt(log(survey_cv(iter) * survey_cv(iter) + extra_survey_cv * extra_survey_cv + 1));
  }
  
  // Calculate vulnerable biomass and U
  Type survey_sex_ratio_nll = 0;
  Type survey_comp_nll = 0;
  Type survey_index_nll = 0;
  Type fishery_comp_nll = 0;
  Type fishery_sex_ratio_nll = 0;
  Type catch_nll = 0.0;
  Type sum1 = 0.0;
  Type sum2 = 0.0;
  
  /*
   * Build Covariance's for Observations and states currently just iid with different covariances
   */
  
  /*
   * Deal with F stuff
   */
  
  for(sex_iter = 0; sex_iter < 2; ++sex_iter) {
    for(age_ndx = 0; age_ndx < nages; ++age_ndx) {
      fishery_selectivity(age_ndx, sex_iter) = Type(1)/(Type(1) + pow(Type(19),((f_a50(sex_iter) - ages(age_ndx))/f_ato95(sex_iter))));
      survey_selectivity(age_ndx, sex_iter) = Type(1)/(Type(1) + pow(Type(19),((survey_a50(sex_iter) - ages(age_ndx))/survey_ato95(sex_iter))));
      if(sex_iter == 1) {
        fishery_selectivity(age_ndx, sex_iter) *= f_alpha_f;
        survey_selectivity(age_ndx, sex_iter) *= survey_alpha_f;
      }
    }
  }
  
  for(year_ndx = 0; year_ndx < nyears; year_ndx++) {
    for(age_ndx = 0; age_ndx < nages; ++age_ndx) {
      for(sex_iter = 0; sex_iter < 2; ++sex_iter) {
        F_ay(age_ndx, year_ndx, sex_iter) = fishery_selectivity(age_ndx, sex_iter) * annual_F(year_ndx);
        Z_ay(age_ndx, year_ndx, sex_iter) = F_ay(age_ndx, year_ndx, sex_iter) + natMor;
      }
    }
  }
   
  /*
   * Initialise first year
   */
  // Initialise Partition
  
  
  for(age_ndx = 0; age_ndx < nages; ++age_ndx) {
    temp_partition_m(age_ndx) = R0 * exp(-(ages(age_ndx)) * natMor) * proportion_male(0);
    temp_partition_f(age_ndx) = R0 * exp(-(ages(age_ndx)) * natMor) * (1 - proportion_male(0));
  }
  if(maxAgePlusGroup == 1) {
    temp_partition_m(nages - 1) = proportion_male(0) * R0 * exp(- ages(nages - 1) * natMor) / (1.0 - exp(-natMor));
    temp_partition_f(nages - 1) = (1 - proportion_male(0)) * R0 * exp(- ages(nages - 1) * natMor) / (1.0 - exp(-natMor));
  }
  // Applying ageing
  Type plus_group_m = temp_partition_m(nages - 1);
  Type plus_group_f = temp_partition_f(nages - 1);
  for(age_ndx = 1; age_ndx < (nages); ++age_ndx) {
    N(age_ndx, 0, 0) =  temp_partition_m(age_ndx - 1);
    N(age_ndx, 0, 1) =  temp_partition_f(age_ndx - 1);
  }
  N(nages - 1, 0, 0) += plus_group_m;
  N(nages - 1, 0, 1) += plus_group_f;
  N(0, 0, 0) = R0 * proportion_male(0);
  N(0, 0, 1) = R0 * (1 - proportion_male(0));
  
  for(age_ndx = 1; age_ndx < (nages - 1); ++age_ndx) {
    ssb(0) += N(age_ndx, 0, 1) * exp(-natMor * propZ_ssb(0))  * stockMeanWeight(age_ndx, 0, 1) * prop_mature(age_ndx, 0);
    ssb(0) += N(age_ndx, 0, 0) * exp(-natMor * propZ_ssb(0))  * stockMeanWeight(age_ndx, 0, 0) * prop_mature(age_ndx, 0);
  }
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
      predlogN(0,0) = log(N(0,year_ndx - 1, 0) * ycs(year_ndx - 1) * proportion_male(year_ndx - 1));
      predlogN(0,1) = log(N(0,year_ndx - 1, 1) * ycs(year_ndx - 1) * (1 - proportion_male(year_ndx - 1)));
      
    } else if(stockRecruitmentModelCode == 2) { //BH
      predlogN(0, 0) = log(R0 * BevertonHolt(ssb(year_ndx - 1), B0, steepness) * ycs(year_ndx - 1) * proportion_male(year_ndx - 1));
      predlogN(0, 1) = log(R0 * BevertonHolt(ssb(year_ndx - 1), B0, steepness) * ycs(year_ndx - 1) * (1 - proportion_male(year_ndx - 1)));
      
    } else{
      error("SR model code not recognized");
    }
    annual_recruitment(year_ndx - 1) = exp(predlogN(0, 0)) + exp(predlogN(0, 1));
    //----------------
    // F + M + ageing
    //----------------
    for(sex_iter = 0; sex_iter < 2; ++sex_iter) {
      for(age_ndx = 1; age_ndx < nages; ++age_ndx) 
        predlogN(age_ndx, sex_iter) = log(N(age_ndx - 1, year_ndx - 1, sex_iter)) - Z_ay(age_ndx - 1, year_ndx - 1, sex_iter);
      
      if(maxAgePlusGroup == 1) {
        predlogN(nages - 1, sex_iter) = log(N(nages - 2, year_ndx - 1, sex_iter) * exp(- Z_ay(nages - 2, year_ndx - 1, sex_iter)) +
          N(nages - 1, year_ndx - 1, sex_iter) * exp(-Z_ay(nages - 1, year_ndx - 1, sex_iter)));
      }
      // Transform from log space
      for(age_ndx = 0; age_ndx < nages; ++age_ndx) 
        N(age_ndx, year_ndx, sex_iter)  = exp(predlogN(age_ndx, sex_iter));
        
    }
    
    // Calculate SSBs an interpolation bewtween the year, starting with previous years Paritition
    for(age_ndx = 0; age_ndx < nages; ++age_ndx) {
      ssb(year_ndx) += N(age_ndx, year_ndx - 1, 1) * exp(-Z_ay(age_ndx, year_ndx - 1, 1) * propZ_ssb(year_ndx - 1)) * prop_mature(age_ndx, year_ndx - 1) * stockMeanWeight(age_ndx, year_ndx - 1, 1);
      ssb(year_ndx) += N(age_ndx, year_ndx - 1, 0) * exp(-Z_ay(age_ndx, year_ndx - 1, 0) * propZ_ssb(year_ndx - 1)) * prop_mature(age_ndx, year_ndx - 1) * stockMeanWeight(age_ndx, year_ndx - 1, 0);
      
      for(sex_iter = 0; sex_iter < 2; ++sex_iter) 
        pred_catches(year_ndx - 1) += exp(log(N(age_ndx, year_ndx - 1, sex_iter)) + log(F_ay(age_ndx, year_ndx - 1, sex_iter)) - log(Z_ay(age_ndx, year_ndx - 1, sex_iter)) + log(1 - exp(-Z_ay(age_ndx, year_ndx - 1, sex_iter)))) * catchMeanWeight(age_ndx, year_ndx - 1, sex_iter);
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
      for(sex_iter = 0; sex_iter < 2; ++sex_iter) {
        for(age_ndx = 0; age_ndx < nages; ++age_ndx) {
          survey_partition(age_ndx, sex_iter) = survey_Q * survey_selectivity(age_ndx, sex_iter) * N(age_ndx, year_ndx,sex_iter) * exp(-(Z_ay(age_ndx, year_ndx,sex_iter)) * propZ_survey(iter));
          survey_index_fitted(iter) += survey_partition(age_ndx, sex_iter) * stockMeanWeight(age_ndx, year_ndx, sex_iter);
          survey_numbers_by_sex(iter, sex_iter) += survey_partition(age_ndx, sex_iter);
        }
      }
      // ageing error and store in fitted
      for(sex_iter = 0; sex_iter < 2; ++sex_iter) {
        temp_partition.fill(0.0);
        for (int a1 = 0; a1 < nages; ++a1) {
          for (int a2 = 0; a2 < nages; ++a2) 
            temp_partition(a2) += survey_partition(a1, sex_iter) * ageing_error_matrix(a1, a2);
        }
        for(age_ndx = 0; age_ndx < nages; ++age_ndx) 
          survey_comp_fitted(age_ndx + (sex_iter * nages), iter) = temp_partition(age_ndx);
      }
      ++iter;
    }
  }
  
  // Fishery 
  iter = 0;
  for(year_ndx = 0; year_ndx < nyears; year_ndx++) {
    if (fishery_year_indicator(year_ndx) == 1) {
      fishery_partition.fill(0.0);
      for(sex_iter = 0; sex_iter < 2; ++sex_iter) {
        for(age_ndx = 0; age_ndx < nages; ++age_ndx) {
          fishery_partition(age_ndx, sex_iter) = N(age_ndx, year_ndx, sex_iter)  * (1 - exp(- Z_ay(age_ndx, year_ndx, sex_iter))) * F_ay(age_ndx, year_ndx, sex_iter) / Z_ay(age_ndx, year_ndx, sex_iter);
          fishery_numbers_by_sex(iter, sex_iter) += fishery_partition(age_ndx, sex_iter);
        }
      }
      // ageing error  
      for(sex_iter = 0; sex_iter < 2; ++sex_iter) {
        temp_partition.fill(0.0);
        for (int a1 = 0; a1 < nages; ++a1) {
          for (int a2 = 0; a2 < nages; ++a2) 
            temp_partition[a2] += fishery_partition(a1, sex_iter) * ageing_error_matrix(a1,a2);
        }
        for(age_ndx = 0; age_ndx < nages; ++age_ndx) 
          fishery_comp_fitted(age_ndx + (sex_iter * nages), iter) = temp_partition(age_ndx);
      }
      ++iter;
    }
  }
  
  // Evaluate ll and simulate if we want to
  iter = 0;
  vector<Type> temp_age_observed(nages);
  for(year_ndx = 0; year_ndx < nyears; year_ndx++) {
    if (survey_year_indicator(year_ndx) == 1) {
      // Prop at age + biomass index
      survey_index_nll -= survey_index_keep(iter) * dnorm(log(survey_obs(iter)), log(survey_index_fitted(iter)) - 0.5 * survey_sd(iter) * survey_sd(iter), survey_sd(iter), true);
      //std::cout << "iter = " << iter << " val = " << survey_index_nll << " lower = " << survey_index_keep.cdf_lower(iter) << " upper = " << survey_index_keep.cdf_upper(iter) << " pnorm = " << log( pnorm(log(survey_obs(iter)), log(survey_index_fitted(iter)) - 0.5 * survey_sd(iter) * survey_sd(iter), survey_sd(iter)) )<< "\n";
      SIMULATE {
        survey_obs(iter) = exp(rnorm(log(survey_index_fitted(iter)) - 0.5 * survey_sd(iter) * survey_sd(iter), survey_sd(iter)));
      }
      if(survey_AF_type == 0) {
        // Sexes independent
        for(sex_iter = 0; sex_iter < 2; ++sex_iter) {
          Type effective_sample_size = 0;
          for (age_ndx = 0; age_ndx < nages; ++age_ndx) {
            effective_sample_size += survey_AF_obs(age_ndx + (sex_iter * nages), iter);
            temp_partition(age_ndx) = survey_comp_fitted(age_ndx + (sex_iter * nages), iter);
          }
          temp_partition /= temp_partition.sum();
          for (age_ndx = 0; age_ndx < nages; ++age_ndx) {
            survey_comp_fitted(age_ndx + (sex_iter * nages), iter) = temp_partition(age_ndx);
            temp_age_observed(age_ndx) = survey_AF_obs(age_ndx + (sex_iter * nages), iter);
          }
          // Multinomial
          survey_comp_nll -= dmultinom(temp_age_observed, temp_partition, true);
          SIMULATE {
            temp_age_observed = rmultinom(temp_partition, effective_sample_size);
            for (age_ndx = 0; age_ndx < nages; ++age_ndx) {
              survey_AF_obs(age_ndx + (sex_iter * nages), iter) = temp_age_observed(age_ndx);
            }
          }
        }
        
        // do the sex ratio component
        survey_proportion_male_fitted(iter) = survey_numbers_by_sex(iter, 0) / (survey_numbers_by_sex(iter, 0) + survey_numbers_by_sex(iter, 1));
        survey_sex_ratio_nll -= dbinom(survey_numbers_male(iter), sum(survey_AF_obs.col(iter).vec()) , survey_proportion_male_fitted(iter), true);
        SIMULATE {
          survey_numbers_male(iter) = rbinom(sum(survey_AF_obs.col(iter).vec()), survey_proportion_male_fitted(iter));
        }
        
      } else {
        // Sexes combined
        survey_comp_fitted.col(iter) /= survey_comp_fitted.col(iter).vec().sum();
        // Multinomial
        survey_comp_nll -= dmultinom(survey_AF_obs.col(iter).vec(), survey_comp_fitted.col(iter).vec(), true);
        SIMULATE {
          Type N_eff = sum(survey_AF_obs.col(iter));
          survey_AF_obs.col(iter) = rmultinom(survey_comp_fitted.col(iter).vec(), N_eff);
          
        }
      }
      ++iter;
    }
  }
   
  
  iter = 0;
  for(year_ndx = 0; year_ndx < nyears; year_ndx++) {
    if (fishery_year_indicator(year_ndx) == 1) {
      if(fishery_AF_type == 0) {
        // Sexes independent
        for(sex_iter = 0; sex_iter < 2; ++sex_iter) {
          Type effective_sample_size = 0;
          for (age_ndx = 0; age_ndx < nages; ++age_ndx) {
            effective_sample_size += fishery_AF_obs(age_ndx + (sex_iter * nages), iter);
            temp_partition(age_ndx) = fishery_comp_fitted(age_ndx + (sex_iter * nages), iter);
          }
          temp_partition /= temp_partition.sum();
          for (age_ndx = 0; age_ndx < nages; ++age_ndx) {
            fishery_comp_fitted(age_ndx + (sex_iter * nages), iter) = temp_partition(age_ndx);
            temp_age_observed(age_ndx) = fishery_AF_obs(age_ndx + (sex_iter * nages), iter);
          }
          // Multinomial
          fishery_comp_nll -= dmultinom(temp_age_observed, temp_partition, true);
          SIMULATE {
            temp_age_observed = rmultinom(temp_partition, effective_sample_size);
            for (age_ndx = 0; age_ndx < nages; ++age_ndx) {
              fishery_AF_obs(age_ndx + (sex_iter * nages), iter) = temp_age_observed(age_ndx);
            }
          }
        }
        // do the sex ratio component
        fishery_proportion_male_fitted(iter) = fishery_numbers_by_sex(iter, 0) / (fishery_numbers_by_sex(iter, 0) + fishery_numbers_by_sex(iter, 1));
        fishery_sex_ratio_nll -= dbinom(fishery_numbers_male(iter), sum(fishery_AF_obs.col(iter).vec()) , fishery_proportion_male_fitted(iter), true);
        SIMULATE {
          fishery_numbers_male(iter) = rbinom(sum(fishery_AF_obs.col(iter).vec()), fishery_proportion_male_fitted(iter));
        }
      } else {
        // Sexes combined
        fishery_comp_fitted.col(iter) /= fishery_comp_fitted.col(iter).vec().sum();
        // Multinomial
        fishery_comp_nll -= dmultinom(fishery_AF_obs.col(iter).vec(), fishery_comp_fitted.col(iter).vec(), true);
        SIMULATE {
          Type N_eff = sum(fishery_AF_obs.col(iter));
          fishery_AF_obs.col(iter) = rmultinom(fishery_comp_fitted.col(iter).vec(), N_eff);
          
        }
      }
      ++iter;
    }
  }
  
  catch_nll =  -1.0 * sum(dnorm(log(catches), log(pred_catches) - 0.5 * catch_sd * catch_sd, catch_sd, true));
  
  SIMULATE {
    vector<Type> log_mean_pred_catches = log(pred_catches) - 0.5 * catch_sd * catch_sd;
    catches = exp(rnorm(log_mean_pred_catches, catch_sd));
    REPORT(survey_AF_obs);
    REPORT(fishery_AF_obs);
    REPORT(catches);
    REPORT( survey_obs );
    REPORT( survey_numbers_male );
    REPORT( fishery_numbers_male );
    
  }
  
  
  Type joint_nll = fishery_comp_nll + survey_comp_nll + survey_index_nll + recruit_nll + catch_nll + survey_sex_ratio_nll;
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
  REPORT(f_alpha_f);
  REPORT(survey_alpha_f);
  
  REPORT( N );
  REPORT( ycs );
  REPORT( annual_recruitment );
  REPORT( survey_a50 );
  REPORT( survey_ato95 ); 
  REPORT( survey_Q );
  REPORT( f_a50 );
  REPORT( f_ato95 );
  REPORT( annual_F );
  REPORT( extra_survey_cv );
  REPORT( survey_comp_fitted );
  REPORT( survey_index_fitted );
  REPORT( survey_index_nll );

  REPORT( catch_sd );
  REPORT( fishery_comp_fitted );

  REPORT( F_ay );  
  REPORT( Z_ay );
  ADREPORT( ssb );
  
  
  REPORT(stockMeanWeight);
  REPORT(catchMeanWeight);
  
  REPORT(survey_proportion_male_fitted); 
  REPORT(survey_numbers_by_sex); 
  
  // ADREPORT(logN);
  // ADREPORT(logF);
  return joint_nll; // we are minimising with nlminb
   
  //return 0.0;
}
