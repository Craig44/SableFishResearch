// SimpleTagEstimator for exploring and testing alternative tag-likelihoods.

#include <TMB.hpp>

// Zerofun functions see - for a discussion on this https://github.com/kaskr/adcomp/issues/7
template<class Type>
Type posfun(Type x, Type eps, Type &pen) {
  pen += CppAD::CondExpLt(x,eps,Type(0.01)*pow(x-eps,2),Type(0));
  Type xp = -(x/eps-1);
  return CppAD::CondExpGe(x,eps,x,
                          eps*(1/(1+xp+pow(xp,2)+pow(xp,3)+pow(xp,4)+pow(xp,5))));
}


/*
 *  an index folding method to help get the folded index given dim1 and dim2
 *  @param ndx_1 (starts at 0 goes to (n_regions - 1))
 *  @param ndx2
 *  @param n_elements_dim1
 *  @return an index to look up the tagged partition which covers both these indicies
 */
int get_folded_ndx(int ndx_1, int ndx2, int n_elements_dim1) {
  return ndx2 * n_elements_dim1 + ndx_1;
}

template<class Type>
Type objective_function<Type>::operator() ()
  {
  /*
   * Declare Namespace
   */
  using namespace density;

  DATA_INTEGER(n_y);                                // number of years to project the model beyond max(years)
  DATA_INTEGER(n_regions);                          // number of regions in the model
  DATA_SCALAR(M);                                   // assumes min(ages) >= 1, also assumes the last age is a plus group
  DATA_VECTOR(F_y);                                 // annual years
  DATA_INTEGER(tag_likelihood);                     // 0 = Poisson, 1 = Multinomial (release-conditioned), 2 = Multinomial (recapture-conditioned), 3 = recapture_conditioned (McGarvey formualtion)
  DATA_ARRAY(tag_recovery_obs);                     // dimension :n_regions x n_y x n_release_events
  DATA_VECTOR(number_of_tags_released);             // n_regions length
  DATA_INTEGER(movement_transformation);            // 0 = simplex, 1=  multinomial logit transformation
  int n_release_events = n_regions;
  array<Type> tag_recovery_expected(tag_recovery_obs.dim);

  PARAMETER_ARRAY(transformed_movement_pars);            // transformed parameters for movmenet (consider both simplex and logistic? or what ever it is). dimension:  (n_regions - 1) x n_regions
  PARAMETER(ln_phi);            // transformed parameters for movmenet (consider both simplex and logistic? or what ever it is). dimension:  (n_regions - 1) x n_regions

  // indicies that we will use consistently

  int year_ndx;
  int region_ndx;
  int release_ndx;
  int recovery_ndx;
  // deal with movement
  matrix<Type> movement_matrix(n_regions,n_regions);                  // n_regions x n_regions. Rows sum = 1 (aka source)
  if(movement_transformation == 0) {
    // simplex
    vector<Type> cache_log_k_value(n_regions - 1);
    for(int k = 0; k < (n_regions - 1); k++)
      cache_log_k_value[k] = log(n_regions - 1 - k);

    for(region_ndx = 0; region_ndx < n_regions; ++region_ndx) {
      Type stick_length = 1.0;
      for (int k = 0; k < (n_regions - 1); ++k) {
        movement_matrix(region_ndx, k) = stick_length * invlogit(transformed_movement_pars(k, region_ndx) - cache_log_k_value(k));
        stick_length -= movement_matrix(region_ndx, k);
      }
      // plus group
      movement_matrix(region_ndx, n_regions - 1) = stick_length;
    }
  } else if(movement_transformation == 1) {
    //  multinomial logit transformation

    Type sum_exp;
    for(region_ndx = 0; region_ndx < n_regions; ++region_ndx) {
      sum_exp = (exp(transformed_movement_pars.col(region_ndx).vec())).sum();
      //std::cerr << "sum_exp " << sum_exp << "\n";
      for (int k = 0; k < (n_regions - 1); ++k)
        movement_matrix(region_ndx, k) = exp(transformed_movement_pars(k, region_ndx)) / (1.0 + sum_exp);
      movement_matrix(region_ndx, n_regions - 1) = 1.0 / (1.0 + sum_exp);
    }
  }

  Type phi = exp(ln_phi);
  Type s1,s2;
  // Start the model
  vector<Type> Z_y(F_y.size());
  Z_y = F_y + M;
  vector<Type> S_y(F_y.size());
  S_y = exp(-1.0 * Z_y);
  Type temp_predict_tag = 0.0;
  Type eps_for_posfun = 0.00001; // used for the posfun object to scale values above zero
  Type nll = 0.0;
  Type posfun_pen = 0.0;
  array<Type> tag_partition(n_regions, n_y + 1, n_release_events);
  tag_partition.fill(0.0);
  vector<Type> obs_release_condition_multinomial_vector(n_y * n_regions + 1); // plus one for the NC group
  vector<Type> pred_release_condition_multinomial_vector(n_y * n_regions + 1); // plus one for the NC group
  vector<Type> obs_recapture_condition_multinomial_vector( n_regions); //
  vector<Type> pred_recapture_condition_multinomial_vector( n_regions); //
  // seed releases
  for(release_ndx = 0; release_ndx < n_release_events; ++release_ndx)
    tag_partition(release_ndx, 0, release_ndx) = number_of_tags_released(release_ndx);
  // generate expected tag-recoveries

  for(year_ndx = 0; year_ndx < n_y; ++year_ndx) {
    for(release_ndx = 0; release_ndx < n_release_events; ++release_ndx) {
      // Z + ageing
      tag_partition.col(release_ndx).col(year_ndx + 1) = tag_partition.col(release_ndx).col(year_ndx) * S_y(year_ndx);
      // movement
      tag_partition.col(release_ndx).col(year_ndx + 1) = (movement_matrix * tag_partition.col(release_ndx).col(year_ndx + 1).matrix()).array();
    }
  }
  vector<Type> total_predicted_releases(n_release_events);
  total_predicted_releases.setZero();
  // Generate expected tag-recoveries
  for(year_ndx = 0; year_ndx < n_y; ++year_ndx) {
    for(release_ndx = 0; release_ndx < n_release_events; ++release_ndx) {
      for(region_ndx = 0; region_ndx < n_regions; ++region_ndx) {
        tag_recovery_expected(region_ndx, year_ndx, release_ndx) = tag_partition(region_ndx, year_ndx + 1, release_ndx) * F_y(year_ndx) / Z_y(year_ndx) * (1.0 - S_y(year_ndx));
        total_predicted_releases(release_ndx) += tag_recovery_expected(region_ndx, year_ndx, release_ndx);
      }
    }
  }

  // Now format them based on the likelihood type.
  if(tag_likelihood == 0) {
    // Poisson
    for(year_ndx = 0; year_ndx < n_y; ++year_ndx) {
      for(release_ndx = 0; release_ndx < n_release_events; ++release_ndx) {
        for(region_ndx = 0; region_ndx < n_regions; ++region_ndx) {
          nll -= dpois(tag_recovery_obs(region_ndx, year_ndx, release_ndx), posfun(tag_recovery_expected(region_ndx, year_ndx, release_ndx), eps_for_posfun, posfun_pen), true);
          SIMULATE {
            tag_recovery_obs(region_ndx, year_ndx, release_ndx) = rpois(tag_recovery_expected(region_ndx, year_ndx, release_ndx));
          }
        }
      }
    }
  } else if(tag_likelihood == 1) {
    // 1 = Multinomial (release-conditioned)
    for(release_ndx = 0; release_ndx < n_release_events; ++release_ndx) {
      pred_release_condition_multinomial_vector.setZero(); // reset this to zeros
      obs_release_condition_multinomial_vector.setZero(); // reset this to zeros
      for(year_ndx = 0; year_ndx < n_y; ++year_ndx) {
        for(region_ndx = 0; region_ndx < n_regions; ++region_ndx) {
          recovery_ndx = get_folded_ndx(region_ndx, year_ndx, n_regions);
          pred_release_condition_multinomial_vector(recovery_ndx) = posfun(tag_recovery_expected(region_ndx, year_ndx, release_ndx), eps_for_posfun, posfun_pen);
          obs_release_condition_multinomial_vector(recovery_ndx) = tag_recovery_obs(region_ndx, year_ndx, release_ndx);
        }
      }
      // scale predicted recaptures by number of tags released
      pred_release_condition_multinomial_vector /= number_of_tags_released(release_ndx);
      // Deal with the non-recpature group
      pred_release_condition_multinomial_vector(pred_release_condition_multinomial_vector.size() - 1) = 1.0 - pred_release_condition_multinomial_vector.sum();
      obs_release_condition_multinomial_vector(obs_release_condition_multinomial_vector.size() - 1) = number_of_tags_released(release_ndx) - obs_release_condition_multinomial_vector.sum();
      // convert predicted to proportions
      // evaluate the likelihood
      nll -= dmultinom(obs_release_condition_multinomial_vector, pred_release_condition_multinomial_vector, true);
    }
  } else if(tag_likelihood == 2) {
    for(release_ndx = 0; release_ndx < n_release_events; ++release_ndx) {
      for(year_ndx = 0; year_ndx < n_y; ++year_ndx) {
        obs_recapture_condition_multinomial_vector = tag_recovery_obs.col(release_ndx).col(year_ndx);
        // need to posfun this
        for(region_ndx = 0; region_ndx < n_regions; ++region_ndx)
          pred_recapture_condition_multinomial_vector(region_ndx) = posfun(tag_recovery_expected(region_ndx, year_ndx, release_ndx), eps_for_posfun, posfun_pen);
        pred_recapture_condition_multinomial_vector /= pred_recapture_condition_multinomial_vector.sum();
        nll -= dmultinom(obs_recapture_condition_multinomial_vector, pred_recapture_condition_multinomial_vector, true);
      }
    }
  } else if(tag_likelihood == 3) {
    for(release_ndx = 0; release_ndx < n_release_events; ++release_ndx) {
      for(year_ndx = 0; year_ndx < n_y; ++year_ndx) {
        obs_recapture_condition_multinomial_vector = tag_recovery_obs.col(release_ndx).col(year_ndx);
        // need to posfun this
        for(region_ndx = 0; region_ndx < n_regions; ++region_ndx) {
          pred_recapture_condition_multinomial_vector(region_ndx) = posfun(tag_recovery_expected(region_ndx, year_ndx, release_ndx), eps_for_posfun, posfun_pen);
          nll -= log(pred_recapture_condition_multinomial_vector(region_ndx)) * obs_recapture_condition_multinomial_vector(region_ndx);
        }
      }
    }
  } else if(tag_likelihood == 4) {
    // Negative binomial
    for(year_ndx = 0; year_ndx < n_y; ++year_ndx) {
      for(release_ndx = 0; release_ndx < n_release_events; ++release_ndx) {
        for(region_ndx = 0; region_ndx < n_regions; ++region_ndx) {
          s1 = log(posfun(tag_recovery_expected(region_ndx, year_ndx, release_ndx), eps_for_posfun, posfun_pen));                          // log(mu)
          s2 = 2. * s1 - ln_phi;                         // log(var - mu)
          nll -= dnbinom_robust(tag_recovery_obs(region_ndx, year_ndx, release_ndx), s1, s2, true);
          SIMULATE {
            tag_recovery_obs(region_ndx, year_ndx, release_ndx) = rnbinom2(s1, s2);
          }
        }
      }
    }
  }
  REPORT(tag_partition);
  REPORT(S_y);
  REPORT(Z_y);
  REPORT(movement_matrix);
  REPORT(tag_recovery_expected);

  REPORT(nll);
  REPORT(posfun_pen);
  REPORT(phi);

  ADREPORT( movement_matrix );

  return nll + posfun_pen;
}
