# Load the setup functions
source("setup_functions.R")

# This R file generates Fig. 13, which demonstrates three parallel experiments with varying misspecification levels of sigma_y,M* = 0.5, 0.7, and 1. 
# The current code uses sigma_y,M* = 1. To use the other levels, simply change the first line.

# Set the misspecification level (sigma_y,M*)
syM_true <- 1  # Change this to 0.5 or 0.7 as needed

# hyper-parameter setting
theta <- theta_true <- 0.5
sigma_x <- sx_true <- 0.5^0.5
syA_true <- 1

# build structure for test data 
test_block <- 100
pattern_test <- c(4, 2, test_block)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
loc_test <- places(pattern_test)
loc_test <-places(pattern_test, missing_locs=which(!loc_test$Ox)) #put in missing once have structure
N_test <- loc_test$N-pattern_test[2]
n_test <- loc_test$n-pattern_test[2]
m_test <- loc_test$m
Ox_test <- loc_test$Ox[(0.5*pattern_test[2]+1):(N_test+0.5*pattern_test[2])]; #true if x is observed
Oy_test <- loc_test$Oy[(0.5*pattern_test[2]+1):(N_test+0.5*pattern_test[2])]; #true if y is observed
nx_test <- ny_test <- n_test; mx_test <- my_test <- m_test


# assign the first ten blocks to be training data
t_block <- 10 
t_index <- 1:(t_block*pattern[2])
t_index_hidden <- 1:(pattern[1]*t_block)

# average over num_data=100 datasets to generate the boxplot
num_data <- 100

# First, get eta estimator for #num_data datasets
## calculate posterior predictive
post_pred_pool_syM1 <- post_pred_prod_syM1 <- matrix(0, length(eta_candidate), num_data)
for(data_num in 1:num_data){
  
  # generate data
  x <- generate_X_sy(data_num, N, theta_true, sx_true)$x
  y <- generate_Y_sy(data_num+1, x, Oy, syA_true, syM_true)$y
  YM <- y[!Oy]
  YA <- y[Oy]
  XA <- x[Ox]
  XA_list <- (1:N)[Ox]
  XM_list <- (1:N)[!Ox]
  
  # build training data
  YA_t <- YA[t_index]
  YM_t <- YM[t_index_hidden]
  XA_t <- XA[t_index]
  XA_list_t <- XA_list[t_index]
  XM_list_t <- XM_list[t_index_hidden]
  
  # build calibration data
  YA_c <- YA[c_index]
  YM_c <- YM[c_index_hidden]
  XA_c <- XA[c_index]
  XA_list_c <- XA_list[c_index]-XA_list[c_index[1]]+1
  XM_list_c <- XM_list[c_index_hidden]-XA_list[c_index[1]]+1
  
  
  for(i_eta in 1:length(eta_candidate)){
      
    fit_eta <- stan("ssm_eta_b.stan", data=list(b=1, eta=eta_candidate[i_eta], N=length(XA_list_t)+length(XM_list_t), nx=length(XA_list_t), mx=length(XM_list_t), ny=length(XA_list_t), my=length(XM_list_t), theta=theta, sigma_x=sigma_x, YA=YA_t, YM=YM_t, XA=XA_t,
                                                YA_list=XA_list_t, YM_list=XM_list_t, XA_list=XA_list_t, XM_list=XM_list_t),
                    iter=main_iter, chains=main_chain, warmup=warmup, thin=thin, refresh=0)
    post_sy2 <- rstan::extract(fit_eta)$sigma2_y
      
    for(c in 1:c_block){
      log_probs <- numeric(length(post_sy2))
      for(t in 1:length(post_sy2)){
        log_probs[t] <- sum(dnorm(YA_c[(2*c-1):(2*c)], XA_c[(2*c-1):(2*c)], post_sy2[t]^0.5, log=TRUE))
      }
      log_sum_t <- log_sum_exp(log_probs)
      post_pred_prod_syM1[i_eta, data_num] <- post_pred_prod_syM1[i_eta, data_num] + log_sum_t
    }
      
    log_p_matrix <- matrix(0, nrow=length(post_sy2), ncol=length(YA_c))
    for(i in 1:length(post_sy2)){
      log_p_matrix[i, ] <- dnorm(YA_c, XA_c, post_sy2[i], log=TRUE)
    }
    post_pred_pool_syM1[i_eta, data_num] <- (log_sum_exp(apply(log_p_matrix, 1, sum)))
  }
}

## get three eta estimates from posterior predictive, i.e., optimal, posterior mean and harmonic mean
eta_est_prod <- eta_est_pool <- matrix(0, num_data, 3)
for(data_num in 1:num_data){
  
  smooth_prod <- smooth.spline(eta_candidate, post_pred_prod_syM1[,data_num], spar=0.3)
  log_px <- predict(smooth_prod, eta_grid)$y
  eta_est_prod[data_num, 1] <- eta_grid[which.max(log_px)]
  eta_post_prod <- update_eta_uniform(10000, 1, 0.1, smooth_prod)$eta
  eta_est_prod[data_num,2] <- mean(eta_post_prod)
  eta_est_prod[data_num,3] <- 1/mean(1/eta_post_prod)
  
  smooth_pool <- smooth.spline(eta_candidate, post_pred_pool_syM1[,data_num], spar=0.3)
  log_px <- predict(smooth_pool, eta_grid)$y
  eta_est_pool[data_num, 1] <- eta_grid[which.max(log_px)]
  eta_post_pool <- update_eta_uniform(10000, 1, 0.1, smooth_pool)$eta
  eta_est_pool[data_num,2] <- mean(eta_post_pool)
  eta_est_pool[data_num,3] <- 1/mean(1/eta_post_pool)
}


# Second, calculate posterior predictive on test data using eta derived from the first step
post_pred_prod_eta_opt_syM1 <- post_pred_prod_eta_pm_syM1 <- post_pred_prod_eta_hm_syM1 <- post_pred_prod_bayes_syM1 <- post_pred_prod_cut_syM1 <- numeric(num_data)
post_pred_pool_eta_opt_syM1 <- post_pred_pool_eta_pm_syM1 <- post_pred_pool_eta_hm_syM1 <- post_pred_pool_bayes_syM1 <- post_pred_pool_cut_syM1 <- numeric(num_data)

for(data_num in 1:num_data){
  
  # generate training+calibration data, same data as the first step
  x <- generate_X_sy(data_num, N, theta_true, sx_true)$x
  y <- generate_Y_sy(data_num+1, x, Oy, syA_true, syM_true)$y
  YM <- y[!Oy]
  YA <- y[Oy]
  XA <- x[Ox]
  XA_list <- (1:N)[Ox]
  XM_list <- (1:N)[!Ox]
  
  # randomly generate a test data
  x_test <- generate_X_sy(sample(1:10000, 1), N_test, theta_true, sx_true)$x
  y_test <- generate_Y_sy(sample(1:10000, 1), x_test, Oy_test, syA_true, syM_true)$y
  YM_test <- y_test[!Oy_test]
  YA_test <- y_test[Oy_test]
  XA_test <- x_test[Ox_test]
  XA_list_test <- (1:N_test)[Ox_test]
  XM_list_test <- (1:N_test)[!Ox_test]
  
  # pooled
  ## eta optimal
  fit <- stan("ssm_eta_b.stan", data=list(b=1, eta=eta_est_pool[data_num,1], N=length(XA_list)+length(XM_list), nx=length(XA_list), mx=length(XM_list), ny=length(XA_list), my=length(XM_list), theta=theta, sigma_x=sigma_x, YA=YA, YM=YM, XA=XA,
                                          YA_list=XA_list, YM_list=XM_list, XA_list=XA_list, XM_list=XM_list),
              iter=main_iter, chains=main_chain, warmup=warmup, thin=thin, refresh=0, seed = as.numeric(Sys.time()))
  post_sy2 <- rstan::extract(fit)$sigma2_y
    
  log_p_matrix <- matrix(0, nrow=length(post_sy2), ncol=length(YA_test))
  for(i in 1:length(post_sy2)){
    log_p_matrix[i, ] <- dnorm(YA_test, XA_test, post_sy2[i], log=TRUE)
  }
  post_pred_pool_eta_opt_syM1 <- (log_sum_exp(apply(log_p_matrix, 1, sum)))
    
  ## eta pm 
  fit <- stan("ssm_eta_b.stan", data=list(b=1, eta=eta_est_pool[data_num,2], N=length(XA_list)+length(XM_list), nx=length(XA_list), mx=length(XM_list), ny=length(XA_list), my=length(XM_list), theta=theta, sigma_x=sigma_x, YA=YA, YM=YM, XA=XA,
                                          YA_list=XA_list, YM_list=XM_list, XA_list=XA_list, XM_list=XM_list),
               iter=main_iter, chains=main_chain, warmup=warmup, thin=thin, refresh=0, seed = as.numeric(Sys.time()))
  post_sy2 <- rstan::extract(fit)$sigma2_y
    
  log_p_matrix <- matrix(0, nrow=length(post_sy2), ncol=length(YA_test))
  for(i in 1:length(post_sy2)){
    log_p_matrix[i, ] <- dnorm(YA_test, XA_test, post_sy2[i], log=TRUE)
  }
  post_pred_pool_eta_pm_syM1 <- (log_sum_exp(apply(log_p_matrix, 1, sum)))
    
  ## eta hm  
  fit <- stan("ssm_eta_b.stan", data=list(b=1, eta=eta_est_pool[data_num,3], N=length(XA_list)+length(XM_list), nx=length(XA_list), mx=length(XM_list), ny=length(XA_list), my=length(XM_list), theta=theta, sigma_x=sigma_x, YA=YA, YM=YM, XA=XA,
                                          YA_list=XA_list, YM_list=XM_list, XA_list=XA_list, XM_list=XM_list),
              iter=main_iter, chains=main_chain, warmup=warmup, thin=thin, refresh=0, seed = as.numeric(Sys.time()))
  post_sy2 <- rstan::extract(fit)$sigma2_y
    
  log_p_matrix <- matrix(0, nrow=length(post_sy2), ncol=length(YA_test))
  for(i in 1:length(post_sy2)){
    log_p_matrix[i, ] <- dnorm(YA_test, XA_test, post_sy2[i], log=TRUE)
  }
  post_pred_pool_eta_hm_syM1 <- (log_sum_exp(apply(log_p_matrix, 1, sum)))
    
  ## bayes (eta=1)
  fit <- stan("ssm_eta_b.stan", data=list(b=1, eta=1, N=length(XA_list)+length(XM_list), nx=length(XA_list), mx=length(XM_list), ny=length(XA_list), my=length(XM_list), theta=theta, sigma_x=sigma_x, YA=YA, YM=YM, XA=XA,
                                          YA_list=XA_list, YM_list=XM_list, XA_list=XA_list, XM_list=XM_list),
              iter=main_iter, chains=main_chain, warmup=warmup, thin=thin, refresh=0, seed = as.numeric(Sys.time()))
  post_sy2 <- rstan::extract(fit)$sigma2_y
    
  log_p_matrix <- matrix(0, nrow=length(post_sy2), ncol=length(YA_test))
  for(i in 1:length(post_sy2)){
    log_p_matrix[i, ] <- dnorm(YA_test, XA_test, post_sy2[i], log=TRUE)
  }
  post_pred_pool_bayes_syM1 <- (log_sum_exp(apply(log_p_matrix, 1, sum)))
  
  for(c in 1:test_block){
    log_probs <- numeric(length(post_sy2))
    for(t in 1:length(post_sy2)){
      log_probs[t] <- sum(dnorm(YA_test[(2*c-1):(2*c)], XA_test[(2*c-1):(2*c)], post_sy2[t]^0.5, log=TRUE))
    }
    log_sum_t <- log_sum_exp(log_probs)
    post_pred_prod_bayes_syM1 <- post_pred_prod_bayes_syM1 + log_sum_t
  }
  
  
  ## cut (eta=0)
  fit <- stan("ssm_eta_b.stan", data=list(b=1, eta=0, N=length(XA_list)+length(XM_list), nx=length(XA_list), mx=length(XM_list), ny=length(XA_list), my=length(XM_list), theta=theta, sigma_x=sigma_x, YA=YA, YM=YM, XA=XA,
                                          YA_list=XA_list, YM_list=XM_list, XA_list=XA_list, XM_list=XM_list),
              iter=main_iter, chains=main_chain, warmup=warmup, thin=thin, refresh=0, seed = as.numeric(Sys.time()))
  post_sy2 <- rstan::extract(fit)$sigma2_y
    
  log_p_matrix <- matrix(0, nrow=length(post_sy2), ncol=length(YA_test))
  for(i in 1:length(post_sy2)){
    log_p_matrix[i, ] <- dnorm(YA_test, XA_test, post_sy2[i], log=TRUE)
  }
  post_pred_pool_cut_syM1 <- (log_sum_exp(apply(log_p_matrix, 1, sum)))
    
  for(c in 1:test_block){
    log_probs <- numeric(length(post_sy2))
    for(t in 1:length(post_sy2)){
      log_probs[t] <- sum(dnorm(YA_test[(2*c-1):(2*c)], XA_test[(2*c-1):(2*c)], post_sy2[t]^0.5, log=TRUE))
    }
    log_sum_t <- log_sum_exp(log_probs)
    post_pred_prod_cut_syM1 <- post_pred_prod_cut_syM1 + log_sum_t
  }
  
  
  # product
  ## eta optimal
  fit <- stan("ssm_eta_b.stan", data=list(b=1, eta=eta_est_prod[data_num,1], N=length(XA_list)+length(XM_list), nx=length(XA_list), mx=length(XM_list), ny=length(XA_list), my=length(XM_list), theta=theta, sigma_x=sigma_x, YA=YA, YM=YM, XA=XA,
                                          YA_list=XA_list, YM_list=XM_list, XA_list=XA_list, XM_list=XM_list),
              iter=main_iter, chains=main_chain, warmup=warmup, thin=thin, refresh=0, seed = as.numeric(Sys.time()))
  post_sy2 <- rstan::extract(fit)$sigma2_y
  
  for(c in 1:test_block){
    log_probs <- numeric(length(post_sy2))
    for(t in 1:length(post_sy2)){
      log_probs[t] <- sum(dnorm(YA_test[(2*c-1):(2*c)], XA_test[(2*c-1):(2*c)], post_sy2[t]^0.5, log=TRUE))
    }
    log_sum_t <- log_sum_exp(log_probs)
    post_pred_prod_eta_opt_syM1 <- post_pred_prod_eta_opt_syM1 + log_sum_t
  }
  
  ## eta pm 
  fit <- stan("ssm_eta_b.stan", data=list(b=1, eta=eta_est_prod[data_num,2], N=length(XA_list)+length(XM_list), nx=length(XA_list), mx=length(XM_list), ny=length(XA_list), my=length(XM_list), theta=theta, sigma_x=sigma_x, YA=YA, YM=YM, XA=XA,
                                          YA_list=XA_list, YM_list=XM_list, XA_list=XA_list, XM_list=XM_list),
              iter=main_iter, chains=main_chain, warmup=warmup, thin=thin, refresh=0, seed = as.numeric(Sys.time()))
  post_sy2 <- rstan::extract(fit)$sigma2_y
  
  for(c in 1:test_block){
    log_probs <- numeric(length(post_sy2))
    for(t in 1:length(post_sy2)){
      log_probs[t] <- sum(dnorm(YA_test[(2*c-1):(2*c)], XA_test[(2*c-1):(2*c)], post_sy2[t]^0.5, log=TRUE))
    }
    log_sum_t <- log_sum_exp(log_probs)
    post_pred_prod_eta_pm_syM1 <- post_pred_prod_eta_pm_syM1 + log_sum_t
  }
  
  ## eta hm  
  fit <- stan("ssm_eta_b.stan", data=list(b=1, eta=eta_est_prod[data_num,3], N=length(XA_list)+length(XM_list), nx=length(XA_list), mx=length(XM_list), ny=length(XA_list), my=length(XM_list), theta=theta, sigma_x=sigma_x, YA=YA, YM=YM, XA=XA,
                                          YA_list=XA_list, YM_list=XM_list, XA_list=XA_list, XM_list=XM_list),
              iter=main_iter, chains=main_chain, warmup=warmup, thin=thin, refresh=0, seed = as.numeric(Sys.time()))
  post_sy2 <- rstan::extract(fit)$sigma2_y
  
  for(c in 1:test_block){
    log_probs <- numeric(length(post_sy2))
    for(t in 1:length(post_sy2)){
      log_probs[t] <- sum(dnorm(YA_test[(2*c-1):(2*c)], XA_test[(2*c-1):(2*c)], post_sy2[t]^0.5, log=TRUE))
    }
    log_sum_t <- log_sum_exp(log_probs)
    post_pred_prod_eta_hm_syM1 <- post_pred_prod_eta_hm_syM1 + log_sum_t
  }
}


# get ratio, ie., \eta-SMI posterior predictive / Bayes (or cut) posterior predictive
# you can generate boxplot based on these ratios
ratio_opt_bayes_pool_syM1 <- exp(post_pred_pool_eta_opt_syM1 - post_pred_pool_bayes_syM1)
ratio_pm_bayes_pool_syM1 <- exp(post_pred_pool_eta_pm_syM1 - post_pred_pool_bayes_syM1)
ratio_hm_bayes_pool_syM1 <- exp(post_pred_pool_eta_hm_syM1 - post_pred_pool_bayes_syM1)

ratio_opt_cut_pool_syM1 <- exp(post_pred_pool_eta_opt_syM1 - post_pred_pool_cut_syM1)
ratio_pm_cut_pool_syM1 <- exp(post_pred_pool_eta_pm_syM1 - post_pred_pool_cut_syM1)
ratio_hm_cut_pool_syM1 <- exp(post_pred_pool_eta_hm_syM1 - post_pred_pool_cut_syM1)

ratio_opt_bayes_prod_syM1 <- exp(post_pred_prod_eta_opt_syM1 - post_pred_prod_bayes_syM1)
ratio_pm_bayes_prod_syM1 <- exp(post_pred_prod_eta_pm_syM1 - post_pred_prod_bayes_syM1)
ratio_hm_bayes_prod_syM1 <- exp(post_pred_prod_eta_hm_syM1 - post_pred_prod_bayes_syM1)

ratio_opt_cut_prod_syM1 <- exp(post_pred_prod_eta_opt_syM1 - post_pred_prod_cut_syM1)
ratio_pm_cut_prod_syM1 <- exp(post_pred_prod_eta_pm_syM1 - post_pred_prod_cut_syM1)
ratio_hm_cut_prod_syM1 <- exp(post_pred_prod_eta_hm_syM1 - post_pred_prod_cut_syM1)


# To generate Fig.13, we need ratio for the other two misspecification levels, which we use _syM0.5 and_syM0.7 to denote.
value_bayes_all_pool <- c(ratio_opt_bayes_pool_syM0.5, ratio_pm_bayes_pool_syM0.5, ratio_hm_bayes_pool_syM0.5,
                          ratio_opt_bayes_pool_syM0.7, ratio_pm_bayes_pool_syM0.7, ratio_hm_bayes_pool_syM0.7,
                          ratio_opt_bayes_pool_syM1, ratio_pm_bayes_pool_syM1, ratio_hm_bayes_pool_syM1)

value_cut_all_pool <- c(ratio_opt_cut_pool_syM0.5, ratio_pm_cut_pool_syM0.5, ratio_hm_cut_pool_syM0.5,
                        ratio_opt_cut_pool_syM0.7, ratio_pm_cut_pool_syM0.7, ratio_hm_cut_pool_syM0.7,
                        ratio_opt_cut_pool_syM1, ratio_pm_cut_pool_syM1, ratio_hm_cut_pool_syM1)

value_bayes_all_prod <- c(ratio_opt_bayes_prod_syM0.5, ratio_pm_bayes_prod_syM0.5, ratio_hm_bayes_prod_syM0.5,
                          ratio_opt_bayes_prod_syM0.7, ratio_pm_bayes_prod_syM0.7, ratio_hm_bayes_prod_syM0.7,
                          ratio_opt_bayes_prod_syM1, ratio_pm_bayes_prod_syM1, ratio_hm_bayes_prod_syM1)

value_cut_all_prod <- c(ratio_opt_cut_prod_syM0.5, ratio_pm_cut_prod_syM0.5, ratio_hm_cut_prod_syM0.5,
                        ratio_opt_cut_prod_syM0.7, ratio_pm_cut_prod_syM0.7, ratio_hm_cut_prod_syM0.7,
                        ratio_opt_cut_prod_syM1, ratio_pm_cut_prod_syM1, ratio_hm_cut_prod_syM1)


# generate boxplot
group_all <- rep(c("A", "B", "C", "D", "E", "F", "G", "H", "I"), each=100)
dev.new()
boxplot(value_cut_all_pool ~ group_all, log="y", xlab="", ylab="", xaxt="n", col=rep(c("red","green","blue"), each=3),cex.lab=1.5, cex.axis=1.5,cex=0.2)
axis(1, at=1:9, labels = FALSE)
abline(h=1, lty=2)
legend("topright", c(expression(sigma[list(y,M)]^"*" == 0.5), expression(sigma[list(y,M)]^"*" == 0.7), expression(sigma[list(y,M)]^"*" == 1)), col=c("red","green","blue"), pch=15, bg="transparent", cex=1.2)
