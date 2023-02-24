// LEP growth model
/*
  * LEP growth model using both age and length data and tag-increment data
*/
  
#include <TMB.hpp>
  
// logistic ogive function parametersised with a_50 and a_to95
template <class Type>
vector<Type> logistic_ogive(vector<Type>& ages, Type& sel_50, Type& sel_95) {
  //std::cout << "logistic_ogive\n";
  int n_ages = ages.size();
  vector<Type> logis(n_ages);
  for (int age = 0;  age < n_ages; ++age) {
    logis[age] = Type(1.0) / (Type(1.0) + pow(Type(19.0), (sel_50 - ages[age]) / sel_95));
  }
  return logis;
}  
// Zerofun functions see - for a discussion on this https://github.com/kaskr/adcomp/issues/7
template<class Type>
Type posfun(Type x, Type eps, Type &pen) {
  pen += CppAD::CondExpLt(x,eps,Type(0.01)*pow(x-eps,2),Type(0));
  Type xp = -(x/eps-1);
  return CppAD::CondExpGe(x,eps,x,
                          eps*(1/(1+xp+pow(xp,2)+pow(xp,3)+pow(xp,4)+pow(xp,5))));
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
template<class Type>
  Type objective_function<Type>::operator() ()
{
  /*
    * Declare Namespace
  */
  using namespace density;
  
  // Input parameters
  DATA_ARRAY(initial_age_tag_releases);             // dim: n_ages x n_regions
  DATA_SCALAR(natural_mortality);                   // natural mortality
  DATA_INTEGER(n_years);
  DATA_INTEGER(n_ages);
  DATA_INTEGER(n_regions);
  DATA_VECTOR(ages);
  DATA_VECTOR(a50_bounds);
  
  DATA_ARRAY(tag_recovery_obs);                     // n_ages x n_release_regions x n_recovery_region x n_recovery_year 
  
  array<Type> young_pred_tag_recovery_obs(tag_recovery_obs.dim[1], tag_recovery_obs.dim[2], tag_recovery_obs.dim[3]);
  array<Type> old_pred_tag_recovery_obs(tag_recovery_obs.dim[1], tag_recovery_obs.dim[2], tag_recovery_obs.dim[3]);
  array<Type> young_obs_tag_recovery_obs(tag_recovery_obs.dim[1], tag_recovery_obs.dim[2], tag_recovery_obs.dim[3]);
  array<Type> old_obs_tag_recovery_obs(tag_recovery_obs.dim[1], tag_recovery_obs.dim[2], tag_recovery_obs.dim[3]);
  
  PARAMETER(logisitic_a50_movement);                               // a50 parameter for age based selectivity movement
  PARAMETER(ln_ato95_movement);                                   // ato95 parameter for age based selectivity movement
  PARAMETER_ARRAY(transformed_movement_pars_young);              // transformed parameters for movmenet (consider both simplex and logistic? or what ever it is). dimension:  (n_regions - 1) x n_regions
  PARAMETER_ARRAY(transformed_movement_pars_old);                // transformed parameters for movmenet (consider both simplex and logistic? or what ever it is). dimension:  (n_regions - 1) x n_regions
  
  // Deal with parameter transformations
  Type ato95_movement = exp(ln_ato95_movement);
  Type a50_movement = invlogit_general(logisitic_a50_movement, a50_bounds(0),a50_bounds(1));
  
  // deal with movement
  // we estimate n-1 parameters for each region based on the simplex transformation
  matrix<Type> movement_matrix_young(n_regions,n_regions);                  // n_regions x n_regions. Rows sum = 1 (aka source)
  matrix<Type> movement_matrix_old(n_regions,n_regions);                  // n_regions x n_regions. Rows sum = 1 (aka source)
  vector<Type> cache_log_k_value(n_regions - 1);
  for(int k = 0; k < (n_regions - 1); k++)
    cache_log_k_value[k] = log(n_regions - 1 - k);
  
  for(int region_ndx = 0; region_ndx < n_regions; ++region_ndx) {
    Type stick_length = 1.0;
    for (int k = 0; k < (n_regions - 1); ++k) {
      movement_matrix_young(region_ndx, k) = stick_length * invlogit(transformed_movement_pars_young(k, region_ndx) - cache_log_k_value(k));
      stick_length -= movement_matrix_young(region_ndx, k);
    }
    // plus group
    movement_matrix_young(region_ndx, n_regions - 1) = stick_length;
  }
  for(int region_ndx = 0; region_ndx < n_regions; ++region_ndx) {
    Type stick_length = 1.0;
    for (int k = 0; k < (n_regions - 1); ++k) {
      movement_matrix_old(region_ndx, k) = stick_length * invlogit(transformed_movement_pars_old(k, region_ndx) - cache_log_k_value(k));
      stick_length -= movement_matrix_old(region_ndx, k);
    }
    // plus group
    movement_matrix_old(region_ndx, n_regions - 1) = stick_length;
  }
  
  Type survivorship = exp(-1.0 * natural_mortality);
  array<Type> tag_numbers_at_age(n_ages, n_regions, n_regions, n_years + 1); // dim: ages x current_region x release_region x years
  matrix<Type> young_natage_for_movement(n_ages, n_regions);        // temp used during movement to account for age-based movement
  matrix<Type> old_natage_for_movement(n_ages, n_regions);          // temp used during movement to account for age-based movement
  Type plus_group;
  vector<Type> young_numbers_at_age(n_ages);
  vector<Type> old_numbers_at_age(n_ages);
  Type pen_posfun = 0; // this is passed to the utility posfun function and added to the likelihood as apenalty
  Type eps_for_posfun = 0.00001; // used for the posfun object to scale values above zero
  Type young_predicted_tags;
  Type old_predicted_tags;
  vector<Type> nll(3);
  nll.setZero();
  // age-based movement ogive
  vector<Type> young_age_based_movement_ogive(n_ages);                       // selectivity for young movement matrix
  vector<Type> old_age_based_movement_ogive(n_ages);                         // selectivity for old movement matrix
  old_age_based_movement_ogive = logistic_ogive(ages, a50_movement, ato95_movement);
  // A max call can cause AD issues.
  Type max_sel = max(old_age_based_movement_ogive);
  for(int age_ndx = 0; age_ndx < n_ages; ++age_ndx)
    old_age_based_movement_ogive(age_ndx) /= max_sel;
  young_age_based_movement_ogive = 1.0 - old_age_based_movement_ogive;
  
  // Initialise
  for(int release_region_ndx = 0; release_region_ndx < n_regions; ++release_region_ndx)
    tag_numbers_at_age.col(0).col(release_region_ndx).col(release_region_ndx) = initial_age_tag_releases.col(release_region_ndx);
  
  /*
   * Annual cycle
   */
  for(int year_ndx = 0; year_ndx < n_years; ++year_ndx) {
    // Movement
    for(int release_region_ndx = 0; release_region_ndx < n_regions; ++release_region_ndx) {
      for(int current_region_ndx = 0; current_region_ndx < n_regions; ++current_region_ndx) {
        young_natage_for_movement.col(current_region_ndx) = tag_numbers_at_age.col(year_ndx).col(release_region_ndx).col(current_region_ndx).vec() * young_age_based_movement_ogive;
        old_natage_for_movement.col(current_region_ndx) = tag_numbers_at_age.col(year_ndx).col(release_region_ndx).col(current_region_ndx).vec() * old_age_based_movement_ogive;
      }
      tag_numbers_at_age.col(year_ndx).col(release_region_ndx) = young_natage_for_movement * movement_matrix_young + old_natage_for_movement * movement_matrix_old;
    }
    
    // Ageing and Z
    for(int release_region_ndx = 0; release_region_ndx < n_regions; ++release_region_ndx) {
      for(int current_region_ndx = 0; current_region_ndx < n_regions; ++current_region_ndx) {
        plus_group = tag_numbers_at_age(n_ages - 1, current_region_ndx, release_region_ndx, year_ndx);
        for(int age_ndx = 0; age_ndx < (n_ages - 1); ++age_ndx) {
          tag_numbers_at_age(age_ndx + 1, current_region_ndx, release_region_ndx, year_ndx + 1) =  tag_numbers_at_age(age_ndx, current_region_ndx, release_region_ndx, year_ndx) * survivorship;
        }
        // plus group
        tag_numbers_at_age(n_ages - 1, current_region_ndx, release_region_ndx, year_ndx + 1) +=  plus_group * survivorship;
      }
    }
    // evaluate tag-observation
    // n_release_regions x n_recovery_region x n_recovery_year 
    for(int release_region_ndx = 0; release_region_ndx < n_regions; ++release_region_ndx) {
      for(int current_region_ndx = 0; current_region_ndx < n_regions; ++current_region_ndx) {
        young_numbers_at_age = tag_numbers_at_age.col(year_ndx).col(release_region_ndx).col(current_region_ndx).vec() * young_age_based_movement_ogive;
        old_numbers_at_age = tag_numbers_at_age.col(year_ndx).col(release_region_ndx).col(current_region_ndx).vec() * old_age_based_movement_ogive;
        
        young_predicted_tags = young_numbers_at_age.sum();
        old_predicted_tags = old_numbers_at_age.sum();
        
        young_predicted_tags = posfun(young_predicted_tags, eps_for_posfun, pen_posfun);
        old_predicted_tags = posfun(old_predicted_tags, eps_for_posfun, pen_posfun);
        
        young_pred_tag_recovery_obs(release_region_ndx, current_region_ndx, year_ndx) = young_predicted_tags;
        old_pred_tag_recovery_obs(release_region_ndx, current_region_ndx, year_ndx) = old_predicted_tags;
        
        // Reformat observations
        young_obs_tag_recovery_obs(release_region_ndx, current_region_ndx, year_ndx) = (tag_recovery_obs.col(year_ndx).col(current_region_ndx).col(release_region_ndx).vec() * young_age_based_movement_ogive).sum();
        old_obs_tag_recovery_obs(release_region_ndx, current_region_ndx, year_ndx) = (tag_recovery_obs.col(year_ndx).col(current_region_ndx).col(release_region_ndx).vec() * old_age_based_movement_ogive).sum();
      
        //
        nll(0) -= dpois(young_obs_tag_recovery_obs(release_region_ndx, current_region_ndx, year_ndx), young_pred_tag_recovery_obs(release_region_ndx, current_region_ndx, year_ndx), true);
        nll(1) -= dpois(old_obs_tag_recovery_obs(release_region_ndx, current_region_ndx, year_ndx), old_pred_tag_recovery_obs(release_region_ndx, current_region_ndx, year_ndx), true);
        
        SIMULATE {
          Type sim_tag_recoveries = rpois(young_pred_tag_recovery_obs(release_region_ndx, current_region_ndx, year_ndx));
          tag_recovery_obs.col(year_ndx).col(current_region_ndx).col(release_region_ndx) = sim_tag_recoveries * (young_numbers_at_age / young_numbers_at_age.sum());
          
          sim_tag_recoveries = rpois(old_pred_tag_recovery_obs(release_region_ndx, current_region_ndx, year_ndx));
          tag_recovery_obs.col(year_ndx).col(current_region_ndx).col(release_region_ndx) += sim_tag_recoveries * (old_numbers_at_age / old_numbers_at_age.sum());
          
        }                                 
      }
    }
  }
  // pos fun penalty for
  nll(2) = pen_posfun;
  
  /*
   * REPORT
   */
  REPORT(movement_matrix_old);
  REPORT(movement_matrix_young);
  REPORT(young_age_based_movement_ogive);
  REPORT(old_age_based_movement_ogive);
  REPORT(tag_numbers_at_age);
  
  REPORT(young_pred_tag_recovery_obs);
  REPORT(old_pred_tag_recovery_obs);
  REPORT(young_obs_tag_recovery_obs);
  REPORT(old_obs_tag_recovery_obs);

  REPORT(tag_recovery_obs);
  
  REPORT( ato95_movement );
  REPORT( a50_movement );
  
  return nll.sum();
  }

