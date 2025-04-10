## load packages
library(rstan)
library(splines)

### functions needed to set up the problem
# determine locations of observed and missing data
places<-function(X_int_pat=c(4,1,10),missing_locs=NA) {
  
  N=X_int_pat[2]+X_int_pat[3]*(X_int_pat[1]+X_int_pat[2])
  n=X_int_pat[2]+X_int_pat[3]*X_int_pat[2]
  m=X_int_pat[3]*X_int_pat[1]
  
  Oy=rep(TRUE,N)
  if (!is.na(missing_locs[1])) {Oy[missing_locs]<-FALSE}
  
  #ObsAt=which(ObsI)
  Ox=c(rep(TRUE,X_int_pat[2]),rep(c(rep(FALSE,X_int_pat[1]),rep(TRUE,X_int_pat[2])),X_int_pat[3]))
  
  return(list(N=N,n=n,m=m,Oy=Oy,Ox=Ox))  
}

# generate hidden states X based on an autoregressive process
generate_X_sy <- function(seed, N, theta_true, sigma_x_true){
  set.seed(seed)
  x <- numeric(N)
  x[1] <- rnorm(1, 0, (sigma_x_true^2/(1-theta_true^2))^0.5)
  for(i in 2:N){
    x[i] <- theta_true*x[i-1] + sigma_x_true*rnorm(1, 0, 1)
  }
  #y <- rnorm(N, x, sigma_y_true)
  
  return(list("x"=x))
}

# generate observed states Y based on hidden states X and observation conditions
generate_Y_sy <- function(seed, x, Oy, syA_true, syM_true){
  y <- numeric(length(x))
  set.seed(seed)
  for(i in 1:length(x)){
    if(Oy[i]){
      y[i] <- rnorm(1, x[i], syA_true)
    }
    else{
      y[i] <- rnorm(1, x[i], syM_true)
    }
  }
  return(list("y"=y))
}

# STAN sampling setting
main_iter <- 2000 # iteration number in the main chain
main_chain <- 4 # number of chains in the first stage
warmup <- 1000
thin <- 1

## sample \eta posterior given loss
update_eta_uniform <- function(main = 1000, T_main = 10, rw_eta = 0.1, model) {
  eta_values <- numeric(main / T_main + 1)
  eta <- eta_values[1] <- runif(1)
 
  for (i in 1:main) {
    eta_new <- rnorm(1, eta, rw_eta)
    if (eta_new >= 0 && eta_new <= 1) {
      r <- (predict(model, eta_new)$y) - (predict(model, eta)$y)
      if (log(runif(1)) < r) {
        eta <- eta_new
      }
    }
   
    if (i %% T_main == 0) {
      eta_values[i / T_main + 1] <- eta
    }
  }
  return(list("eta" = eta_values))
}


# Define the pattern for the problem setup (number of missing states, known states, and blocks)
pattern <- c(4, 2, 20)  # 4 missing, 2 known anchors, 20 blocks

# Data structure setup
loc <- places(pattern)
loc <- places(pattern, missing_locs = which(!loc$Ox))  # Adjust for missing once structure is defined

# Extract the problem parameters based on the pattern
N <- loc$N - pattern[2]  # Total number of states 
n <- loc$n - pattern[2]  # Total number of known states 
m <- loc$m  # Total number of missing states (X_M)
Ox <- loc$Ox[(0.5 * pattern[2] + 1):(N + 0.5 * pattern[2])]  # location of hidden states. TRUE means X_A and FALSE means X_M.
Oy <- loc$Oy[(0.5 * pattern[2] + 1):(N + 0.5 * pattern[2])]  # location of observed states. TRUE means Y_A and FALSE means Y_M.
nx <- ny <- n  # Number of known hidden states (X_A) and known observations (Y_A)
mx <- my <- m  # Number of missing hidden states (X_M) and missing observations (Y_M)
