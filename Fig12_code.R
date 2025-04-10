# Load the setup functions
source("setup_functions.R")

# This R file generates Fig. 12, which demonstrates three parallel experiments with varying misspecification levels of sigma_y,M* = 0.5, 0.7, and 1. 
# The current code uses sigma_y,M* = 1. To use the other levels, simply change the first line.

# Set the misspecification level (sigma_y,M*)
syM_true <- 1  # Change this to 0.5 or 0.7 as needed

# hyper-parameter setting
theta <- theta_true <- 0.5
sigma_x <- sx_true <- 0.5^0.5
syA_true <- 1

## generate a dataset
x <- generate_X_sy(3, N, theta_true, sx_true)$x
y <- generate_Y_sy(7, x, Oy, syA_true, syM_true)$y
YM <- y[!Oy]
YA <- y[Oy]
XA <- x[Ox]
XA_list <- (1:N)[Ox]
XM_list <- (1:N)[!Ox]

## assign the first ten blocks to be training data
t_block <- 10 
t_index <- 1:(t_block*pattern[2])
t_index_hidden <- 1:(pattern[1]*t_block)
YA_t <- YA[t_index]
YM_t <- YM[t_index_hidden]
XA_t <- XA[t_index]
XA_list_t <- XA_list[t_index]
XM_list_t <- XM_list[t_index_hidden]


# candidate \eta values
eta_candidate <- seq(from=0, to=1, by=0.1)

# proportion (# of calibration) / (# of training)
rho_candidate <- c(0.5, 1, 2, 10)

# save posterior predictive value (product and pooled). 
post_pred_prod_syM1 <- post_pred_pool_syM1 <- matrix(0, length(eta_candidate), length(rho_candidate))

# Laplace approximation of posterior predictive using MLE
density_set <- matrix(0, length(eta_candidate), length(rho_candidate))

# pseudo true value for eta is the value that maximizes density at pseudo true value for \sigma_y^2
density_at_truth <- numeric(length(eta_candidate))

# pseudo true value for sigma_y^2
sy_true <- ((n*syA_true^2 + m*syM_true^2)/(n+m))^0.5

# helper function for numerical stability
log_sum_exp <- function(x) {
  m <- max(x)  # Find the maximum value in x
  return(m + log(sum(exp(x - m))))  # Log-sum-exp trick
}


for(i_eta in 1:length(eta_candidate)){
  #print(i_eta)
  fit_eta <- stan("ssm_eta_b.stan", data=list(b=1, eta=eta_candidate[i_eta], N=length(XA_list_t)+length(XM_list_t), nx=length(XA_list_t), mx=length(XM_list_t), ny=length(XA_list_t), my=length(XM_list_t), theta=theta, sigma_x=sigma_x, YA=YA_t, YM=YM_t, XA=XA_t,
                                              YA_list=XA_list_t, YM_list=XM_list_t, XA_list=XA_list_t, XM_list=XM_list_t),
                  iter=main_iter, chains=main_chain, warmup=warmup, thin=thin, refresh=0)
  post_sy2 <- rstan::extract(fit_eta)$sigma2_y
  
  kde <- density(post_sy2)
  density_at_truth[i_eta] <- approx(kde$x, kde$y, xout=sy_true)$y
  if(is.na(density_at_truth[i_eta])){density_at_truth[i_eta] <- 0}
  
  for(i_rho in 1:length(rho_candidate)){
    
    rho <- rho_candidate[i_rho]
    c_block <- rho*t_block  # calibration blocks
    c_index <- (t_block*pattern[2]+1):(pattern[2]*(t_block+c_block))
    c_index_hidden <- (pattern[1]*t_block+1):(pattern[1]*(t_block+c_block))
    YA_c <- YA[c_index]
    YM_c <- YM[c_index_hidden]
    XA_c <- XA[c_index]
    XA_list_c <- XA_list[c_index]-XA_list[c_index[1]]+1
    XM_list_c <- XM_list[c_index_hidden]-XA_list[c_index[1]]+1
    sy2_mle <- mean((YA_c-XA_c)^2)
    
    ## product loss
    for(c in 1:c_block){
      log_probs <- numeric(length(post_sy2))
      for(t in 1:length(post_sy2)){
        log_probs[t] <- sum(dnorm(YA_c[(2*c-1):(2*c)], XA_c[(2*c-1):(2*c)], post_sy2[t]^0.5, log=TRUE))
      }
      log_sum_t <- log_sum_exp(log_probs)
      post_pred_prod_syM1[i_eta, i_rho] <- post_pred_prod_syM1[i_eta, i_rho] + log_sum_t
    }
    
    ## pooled loss
    log_p_matrix <- matrix(0, nrow=length(post_sy2), ncol=length(YA_c))
    for(i in 1:length(post_sy2)){
      log_p_matrix[i, ] <- dnorm(YA_c, XA_c, post_sy2[i], log=TRUE)
    }
    post_pred_pool_syM1[i_eta, i_rho] <- (log_sum_exp(apply(log_p_matrix, 1, sum)))
    
    
    density_set[i_eta, i_rho] <- approx(kde$x, kde$y, xout = sy2_mle)$y
    if(is.na(density_set[i_eta, i_rho])){density_set[i_eta, i_rho] <- 0}
  }
}


## process the result and prepare data for the plot. Basically, smoothing.
eta_grid <- seq(from=0, to=1, by=0.01)
normal_prod <- normal_pool <- normal_laplace <- matrix(0, length(eta_grid), length(rho_candidate))
eta_opt_prod <- eta_pm_prod <- eta_hm_prod <- eta_opt_pool <- eta_pm_pool <- eta_hm_pool <- numeric(length(rho_candidate))

for(i_rho in 1:length(rho_candidate)){
  # product loss model
  log_model_prod <- smooth.spline(eta_candidate, post_pred_prod_syM1[,i_rho], spar=0.3)
  log_px <- predict(log_model_prod, eta_grid)$y
  max_log_px <- max(log_px)
  normal_prod[,i_rho] <- exp(log_px-max_log_px)/(sum(exp(log_px-max_log_px))*(eta_grid[2]-eta_grid[1]))
  eta_opt_prod[i_rho] <- eta_grid[which.max(normal_prod[,i_rho])]
  post_prod <- update_eta_uniform(10000, 10, 0.1, log_model_prod)
  eta_pm_prod[i_rho] <- mean(post_prod$eta)
  eta_hm_prod[i_rho] <- 1/(mean(1/post_prod$eta))
  
  # pooled loss model
  log_model_pool <- smooth.spline(eta_candidate, post_pred_pool_syM1[,i_rho], spar=0.3)
  log_px <- predict(log_model_pool, eta_grid)$y
  max_log_px <- max(log_px)
  normal_pool[,i_rho] <- exp(log_px-max_log_px)/(sum(exp(log_px-max_log_px))*(eta_grid[2]-eta_grid[1]))
  eta_opt_pool[i_rho] <- eta_grid[which.max(normal_pool[,i_rho])]
  post_pool <- update_eta_uniform(10000, 10, 0.1, log_model_pool)
  eta_pm_pool[i_rho] <- mean(post_pool$eta)
  eta_hm_pool[i_rho] <- 1/(mean(1/post_pool$eta))
  
  # Laplace approximation model
  smooth_density <- smooth.spline(eta_candidate, log(density_set[,i_rho]), spar=0.3)
  log_px <- predict(smooth_density, eta_grid)$y
  max_log_px <- max(log_px)
  normal_laplace[,i_rho] <- exp(log_px-max_log_px)/(sum(exp(log_px-max_log_px))*(eta_grid[2]-eta_grid[1]))
}


## pseudo truth of eta
density_smooth <- smooth.spline(eta_candidate, density_at_truth, spar=0.3)
pseudo <- eta_grid[which.max(predict(density_smooth, eta_grid)$y)]


### product 
dev.new()
plot(eta_grid, normal_prod[,1], type="l", xlab="", ylab="", cex.lab=1.5, cex.axis=1.5, lwd=1.5)
lines(eta_grid, normal_prod[,2], col="blue", lwd=1.5)
lines(eta_grid, normal_prod[,3], col="green", lwd=1.5)
lines(eta_grid, normal_prod[,4], col="red", lwd=1.5)
points(eta_pm_prod[1], normal_prod[eta_pm_prod[1]*100+1, 1], pch=25)
points(eta_pm_prod[2], normal_prod[eta_pm_prod[2]*100+1, 2], pch=25, col="blue")
points(eta_pm_prod[3], normal_prod[eta_pm_prod[3]*100+1, 3], pch=25, col="green")
points(eta_pm_prod[4], normal_prod[eta_pm_prod[4]*100+1, 4], pch=25, col="red")
points(eta_hm_prod[1], normal_prod[eta_hm_prod[1]*100+1, 1], pch=4)
points(eta_hm_prod[2], normal_prod[eta_hm_prod[2]*100+1, 2], pch=4, col="blue")
points(eta_hm_prod[3], normal_prod[eta_hm_prod[3]*100+1, 3], pch=4, col="green")
points(eta_hm_prod[4], normal_prod[eta_hm_prod[4]*100+1, 4], pch=4, col="red")
abline(v=pseudo, lty=2)
legend("topleft", c("10", "2", "1", "0.5"), col=c("black", "blue", "green", "red"), pch=20)

### pooled
dev.new()
plot(eta_grid, normal_pool[,1], type="l", xlab="", ylab="", cex.lab=1.5, cex.axis=1.5, lwd=1.5)
lines(eta_grid, normal_pool[,2], col="blue", lwd=1.5)
lines(eta_grid, normal_pool[,3], col="green", lwd=1.5)
lines(eta_grid, normal_pool[,4], col="red", lwd=1.5)
points(eta_pm_pool[1], normal_pool[eta_pm_pool[1]*100+1, 1], pch=25)
points(eta_pm_pool[2], normal_pool[eta_pm_pool[2]*100+1, 2], pch=25, col="blue")
points(eta_pm_pool[3], normal_pool[eta_pm_pool[3]*100+1, 3], pch=25, col="green")
points(eta_pm_pool[4], normal_pool[eta_pm_pool[4]*100+1, 4], pch=25, col="red")
points(eta_hm_pool[1], normal_pool[eta_hm_pool[1]*100+1, 1], pch=4)
points(eta_hm_pool[2], normal_pool[eta_hm_pool[2]*100+1, 2], pch=4, col="blue")
points(eta_hm_pool[3], normal_pool[eta_hm_pool[3]*100+1, 3], pch=4, col="green")
points(eta_hm_pool[4], normal_pool[eta_hm_pool[4]*100+1, 4], pch=4, col="red")
lines(eta_grid, normal_laplace[,1], lty=2)
lines(eta_grid, normal_laplace[,2], col="blue", lty=2)
lines(eta_grid, normal_laplace[,3], col="green", lty=2)
lines(eta_grid, normal_laplace[,4], col="red", lty=2)
abline(v=pseudo, lty=2)
legend("topleft", c("10", "2", "1", "0.5"), col=c("black", "blue", "green", "red"), pch=20)






