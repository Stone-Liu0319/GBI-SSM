data {
  real<lower=0> b;                      // index for beta-loss (b=1/beta)
  real<lower=0, upper=1> eta;           // learning rate 
  int<lower=0> N;                       // Total number of time points (N = n_A + n_M), the total number of states
  int<lower=0> nx;                      // Number of known hidden states (n_A)
  int<lower=0> mx;                      // Number of missing hidden states (n_M)
  int<lower=0> ny;                      // Number of known observed states (Y_A)
  int<lower=0> my;                      // Number of missing observed states (Y_M)
  real<lower=-1, upper=1> theta;        // AR(1) parameter, controls the correlation of the hidden states (X)
  real<lower=0> sigma_x;                // Standard deviation of the AR(1) process for X (the hidden states)
  vector[ny] YA;                        // Known observed states corresponding to known hidden states (Y_A)
  vector[my] YM;                        // Missing observed states corresponding to missing hidden states (Y_M)
  vector[nx] XA;                        // Known hidden states (X_A)
  int YA_list[ny];                      // List that maps the indices of YA to the indices of X_A
  int YM_list[my];                      // List that maps the indices of YM to the indices of X_M
  int XA_list[nx];                      // List that maps the indices of XA to the indices of X_A
  int XM_list[mx];                      // List that maps the indices of XM to the indices of X_M
}

parameters {
  vector[mx] XM;                        // Missing hidden states (X_M), to be inferred
  real<lower=0> sigma2_y;               // Variance of the observation noise (sigma_y^2), to be inferred
}

transformed parameters {
  vector[N] X;                          // The full set of hidden states (X), including both known (X_A) and missing (X_M)
  real ym_loss;                         // Loss term associated with the missing observed states (YM)

  ym_loss = 0;                          // Initialize ym_loss to 0

  // Assign the known hidden states (X_A) to their respective positions in X
  for (i in 1:nx) {
    X[XA_list[i]] = XA[i];              // Known hidden states are directly assigned from XA
  }

  // Assign the missing hidden states (X_M) to their respective positions in X
  for (j in 1:mx) {
    X[XM_list[j]] = XM[j];              // Missing hidden states are inferred from XM
  }

  // Loss term calculation for the missing observed states (YM) when b is not 0 or 1
  if ((b != 0) && (b != 1)){
    for(i in 1:my){
      ym_loss = ym_loss - (exp(-0.5*(YM[i]-X[YM_list[i]])^2/(sigma2_y)) / ((2*pi())^0.5*sigma2_y^0.5))^(1/b-1) / (1/b-1) 
                + (1/b)^(-1.5)*(2*pi())^(-0.5/b+0.5)*sigma2_y^(-0.5/b+0.5);
      // This complex expression computes a modified likelihood for YM when b is not 0 or 1, based on the given formula
    }
  }

  // Special case when b == 0 (i.e., no loss is added for YM)
  if (b == 0) {
    ym_loss = 0;                        // No modification to the likelihood of YM when b == 0
  }
}

model {
  // Prior distributions
  target += inv_gamma_lpdf(sigma2_y | 2, 1);  // Prior for sigma_y^2: Inverse Gamma distribution with shape=2, scale=1
  target += normal_lpdf(X[1] | 0, sqrt(sigma_x^2 / (1 - theta^2))); // Prior for the first hidden state X[1] assuming AR(1)
  
  // AR(1) prior for the rest of the hidden states (X)
  for (i in 2:N) {
    target += normal_lpdf(X[i] | theta * X[i - 1], sigma_x);  // Conditional prior: X[i] ~ N(theta * X[i-1], sigma_x)
  }

  // Likelihood for known observed states (Y_A)
  for(i in 1:ny) {
    target += normal_lpdf(YA[i] | X[YA_list[i]], sqrt(sigma2_y));  // Observed data likelihood for YA given X_A and sigma_y^2
  }

  // For the eta-beta-SMI model: Add the effect of YM if b != 1 (modified likelihood for YM with the regularization term)
  if (b != 1){
    target += -eta * ym_loss;    // Multiply the ym_loss by eta and subtract from the target to account for the regularization
  }

  // For the eta-SMI model: Normal likelihood for YM when b == 1
  if (b == 1) {
    for(j in 1:my) {
      target += eta * normal_lpdf(YM[j] | X[YM_list[j]], sqrt(sigma2_y));  // Likelihood for YM when b == 1
    }
  }
}
