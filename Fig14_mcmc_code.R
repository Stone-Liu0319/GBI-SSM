# Load the setup functions
source("setup_functions.R")

# This file contains MCMC code to sample from (eta, b)-posterior, for pooled loss and product loss, respectively.

## load package
install.packages("invgamma")
library(invgamma)
library(MASS)

### pooled loss
# Prior distribution for sy2 using inverse gamma
prior_sy2 <- function(sy2) {
  return(dinvgamma(sy2, 2, 1, log = TRUE))
}

# Uniform prior for eta and b
prior_eta_b <- function(eta, b) {
  return(dunif(eta, 0, 1, log = TRUE) + dunif(b, 0, 3, log = TRUE))
}

# Main MCMC function 
main_mcmc_pool <- function(n_iter, thin, YA_t, YM_t, XA_t, XA_list_t, XM_list_t, YA_c, XA_c, eta_init=0.5, b_init=1, sy2_init=1, rw_eta=0.1, rw_b=0.5, side_chain) {
  
  # Initialize variables
  eta <- eta_init
  b <- b_init
  sy2 <- sy2_init
  
  # Storage for samples
  samples <- matrix(NA, n_iter / thin, 3)
  colnames(samples) <- c("eta", "b", "phi")
  
  # MCMC sampling loop
  for (t in 1:n_iter) {
    
    # Update eta
    eta_new <- rnorm(1, eta, rw_eta)
    if (eta_new >= 0 & eta_new <= 1) {
      fit_eta <- stan("ssm_eta_b.stan", 
                      data = list(b = b, eta = eta_new, 
                                  N = length(XA_list_t) + length(XM_list_t), 
                                  nx = length(XA_list_t), mx = length(XM_list_t), 
                                  ny = length(XA_list_t), my = length(XM_list_t), 
                                  theta = theta, sigma_x = sigma_x, 
                                  YA = YA_t, YM = YM_t, XA = XA_t,
                                  YA_list = XA_list_t, YM_list = XM_list_t, 
                                  XA_list = XA_list_t, XM_list = XM_list_t),
                      iter = side_chain, chains = 1, refresh = 0)
      sy2_new <- rstan::extract(fit_eta)$sigma2_y[side_chain/2]
      
      # Compute the acceptance ratio for eta
      r <- prior_eta_b(eta_new, b) + sum(dnorm(YA_c, XA_c, sy2_new, log = TRUE)) - 
        prior_eta_b(eta, b) - sum(dnorm(YA_c, XA_c, sy2, log = TRUE))
      
      # Accept or reject eta
      if (log(runif(1)) < r) {
        eta <- eta_new
        sy2 <- sy2_new
      }
    }
    
    # Update b
    b_new <- rnorm(1, b, rw_b)
    if (b_new >= 0 & b_new <= 3) {
      fit_b <- stan("ssm_eta_b.stan", 
                    data = list(b = b_new, eta = eta, 
                                N = length(XA_list_t) + length(XM_list_t), 
                                nx = length(XA_list_t), mx = length(XM_list_t), 
                                ny = length(XA_list_t), my = length(XM_list_t), 
                                theta = theta, sigma_x = sigma_x, 
                                YA = YA_t, YM = YM_t, XA = XA_t,
                                YA_list = XA_list_t, YM_list = XM_list_t, 
                                XA_list = XA_list_t, XM_list = XM_list_t),
                    iter = side_chain, chains = 1, refresh = 0)
      sy2_new <- rstan::extract(fit_b)$sigma2_y[side_chain/2]
      
      # Compute the acceptance ratio for b
      r <- prior_eta_b(eta, b_new) + sum(dnorm(YA_c, XA_c, sy2_new, log = TRUE)) - 
        prior_eta_b(eta, b) - sum(dnorm(YA_c, XA_c, sy2, log = TRUE))
      
      # Accept or reject b
      if (log(runif(1)) < r) {
        b <- b_new
        sy2 <- sy2_new
      }
    }
    
    # Store samples at the specified thinning interval
    if (t %% thin == 0) {
      print(t)
      samples[t / thin, ] <- c(eta, b, sy2)
    }
  }
  
  # Return the collected samples
  return(list("eta"=samples[,1], "b"=samples[,2]))
}


###############################
### product loss
# Log-likelihood function for the given data and parameters (for each block)
log_llk_YAc_prod <- function(YA_c, XA_c, sy2_B, block) {
  
  # the variable "block" is number of Y_A in one block
  
  n_block <- length(YA_c) / block
  log_llk <- 0
  
  # Loop over each block to calculate the log-likelihood
  for (bc in 1:n_block) {
    log_llk <- log_llk + sum(dnorm(YA_c[(block * bc - block + 1):(block * bc)], 
                                   XA_c[(block * bc - block + 1):(block * bc)], 
                                   sy2_B[bc]^0.5, log = TRUE))
  }
  return(log_llk)
}

# MCMC function for sampling eta, b, and phi for each block
mcmc_eta_b_prod <- function(n_iter, thin, YA_t, YM_t, XA_t, XA_list_t, XM_list_t, YA_c, XA_c, 
                            eta_init=0.5, b_init=1, sy2_init=1, xm_init=0.5, rw_eta=0.1, rw_b=0.5, block, side_chain) {
  
  n_block <- length(YA_c) / block
  eta <- eta_init
  b <- b_init
  sy2_B <- rep(sy2_init, n_block)
  xm_B <- matrix(xm_init, length(YM_t), n_block)
  
  # Results matrix to store samples
  results <- matrix(NA, n_iter / thin, 2 + n_block)
  colnames(results) <- c("eta", "b", paste("phi", 1:n_block))
  
  # Acceptance counts for eta and b
  count_accept_eta <- 0
  count_accept_b <- 0
  
  # MCMC iterations
  for (t in 1:n_iter) {
    
    # Update eta
    eta_new <- rnorm(1, eta, rw_eta)
    if (eta_new >= 0 & eta_new <= 1) {
      
      sy2_B_new <- numeric(n_block)
      
      # Fit the model for the new eta
      fit_eta <- stan("ssm_eta_b.stan", 
                      data = list(b = b, eta = eta_new, 
                                  N = length(XA_list_t) + length(XM_list_t), 
                                  nx = length(XA_list_t), mx = length(XM_list_t), 
                                  ny = length(XA_list_t), my = length(XM_list_t), 
                                  theta = theta, sigma_x = sigma_x, 
                                  YA = YA_t, YM = YM_t, XA = XA_t,
                                  YA_list = XA_list_t, YM_list = XM_list_t, 
                                  XA_list = XA_list_t, XM_list = XM_list_t),
                      iter = side_chain, chains = n_block, refresh = 0, 
                      seed = as.numeric(Sys.time()))
      
      # Extract samples and calculate sy2_B_new
      samples <- rstan::extract(fit_eta, permuted = FALSE)
      sy2_B_new <- samples[side_chain/2, , "sigma2_y"]
      names(sy2_B_new) <- NULL
      
      # Calculate acceptance ratio
      r <- log_llk_YAc_prod(YA_c, XA_c, sy2_B_new, block) - log_llk_YAc_prod(YA_c, XA_c, sy2_B, block)
      if (log(runif(1)) < r) {
        count_accept_eta <- count_accept_eta + 1
        eta <- eta_new
        sy2_B <- sy2_B_new
      }
    }
    
    # Update b
    b_new <- rnorm(1, b, rw_b)
    if (b_new >= 0 & b_new <= 3) {
      
      sy2_B_new <- numeric(n_block)
      
      # Fit the model for the new b
      fit_b <- stan("ssm_eta_b.stan", 
                    data = list(b = b_new, eta = eta, 
                                N = length(XA_list_t) + length(XM_list_t), 
                                nx = length(XA_list_t), mx = length(XM_list_t), 
                                ny = length(XA_list_t), my = length(XM_list_t), 
                                theta = theta, sigma_x = sigma_x, 
                                YA = YA_t, YM = YM_t, XA = XA_t,
                                YA_list = XA_list_t, YM_list = XM_list_t, 
                                XA_list = XA_list_t, XM_list = XM_list_t),
                    iter = side_chain, chains = n_block, refresh = 0, 
                    seed = as.numeric(Sys.time()))
      
      # Extract samples and calculate sy2_B_new
      samples <- rstan::extract(fit_b, permuted = FALSE)
      sy2_B_new <- samples[side_chain/2, , "sigma2_y"]
      names(sy2_B_new) <- NULL
      
      # Calculate acceptance ratio
      r <- log_llk_YAc_prod(YA_c, XA_c, sy2_B_new, block) - log_llk_YAc_prod(YA_c, XA_c, sy2_B, block)
      if (log(runif(1)) < r) {
        count_accept_b <- count_accept_b + 1
        b <- b_new
        sy2_B <- sy2_B_new
      }
    }
    
    # Store results at specified thinning interval
    if (t %% thin == 0) {
      print(t)
      results[t / thin, ] <- c(eta, b, sy2_B)
    }
  }
  
  # Return results and acceptance rates
  return(list("results" = results, 
              "accp_eta" = count_accept_eta / n_iter, 
              "accp_b" = count_accept_b / n_iter))
}



######################
# generate data and illustrate how to run MCMC

syM_true <- 1 
theta <- theta_true <- 0.5
sigma_x <- sx_true <- 0.5^0.5
syA_true <- 1

x <- generate_X_sy(3, N, theta_true, sx_true)$x
y <- generate_Y_sy(7, x, Oy, syA_true, syM_true)$y
YM <- y[!Oy]
YA <- y[Oy]
XA <- x[Ox]
XA_list <- (1:N)[Ox]
XM_list <- (1:N)[!Ox]

t_block <- 10 # training blocks
t_index <- 1:(t_block*pattern[2])
t_index_hidden <- 1:(pattern[1]*t_block)
c_block <- 10 # calibration blocks
c_index <- (t_block*pattern[2]+1):(pattern[2]*(t_block+c_block))
c_index_hidden <- (pattern[1]*t_block+1):(pattern[1]*(t_block+c_block))
YA_t <- YA[t_index]
YM_t <- YM[t_index_hidden]
XA_t <- XA[t_index]
XA_list_t <- XA_list[t_index]
XM_list_t <- XM_list[t_index_hidden]
YA_c <- YA[c_index]
YM_c <- YM[c_index_hidden]
XA_c <- XA[c_index]
XA_list_c <- XA_list[c_index]-XA_list[c_index[1]]+1
XM_list_c <- XM_list[c_index_hidden]-XA_list[c_index[1]]+1

## run code and plot contour
result_pool <- main_mcmc_pool(n_iter=10000, thin=1, YA_t, YM_t, XA_t, XA_list_t, XM_list_t, YA_c, XA_c, eta_init=0.5, b_init=0.5, sy2_init=0.5, rw_eta=0.5, rw_b=0.5, side_chain=200)
density_pool <- kde2d(result_pool$eta, result_pool$b)
filled.contour(density_pool, color.palette = terrain.colors)

result_prod <- mcmc_eta_b_prod(n_iter=10000, thin=1, YA_t, YM_t, XA_t, XA_list_t, XM_list_t, YA_c, XA_c, eta_init=0.5, b_init=1, sy2_init=0.5, xm_init=0.5, rw_eta=0.2, rw_b=0.5, block=2, side_chain=200)
density_prod <- kde2d(result_prod$results[,1], result_pool$results[,2])
filled.contour(density_prod, color.palette = terrain.colors)
