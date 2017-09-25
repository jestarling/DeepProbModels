# Fitting a discrete Gaussian mixture model by variational Bayes
# SVI, adapted from James's code.

# utility for equal-variance Gaussian mixture
dnormix = function(x, mu, w) {
	K = length(mu)
	out = rep(0, length(x))
	for(j in 1:K) {
		out = out + w[j]*dnorm(x, mu[j], 1)
	}
	out
}

# constructs a matrix by repeating a vector x over n rows
# like repmat in Matlab
rep_rows = function(x, n) {
	t(matrix(rep(x, n), ncol=n))
}

# hyperpars and sample size
K = 5
tau = 3
N = 1000
minibatch=100

# Mixture model parameters
p_true = rep(1/K, K)
mu_true = rnorm(K, 0, 3)
c_true = sample(K, size=N, replace=TRUE, prob=p_true)
y = rnorm(N, mu_true[c_true], 1)

hist(y, 50)


# initialize variational parameters
m_hat = rnorm(K, 0, 1)
s2_hat = rgamma(K, 2, 2)
full_phi = matrix(1/K,nrow=N, ncol=K)

max_steps = 1000
rel_tol = 1e-9
ELBO_tracker = rep(NA, max_steps)

converged = FALSE
step_counter = 1
old_ELBO = -1e100

# step size parameters
forget = .75
delay = 2.0

while({!converged} & {step_counter <= max_steps}) {
	idx = sample.int(N,size=minibatch)
  
	# update variational distribution for cluster indicators
	if(minibatch >0){
	  psi_hat = outer(y[idx],m_hat) - (0.5* rep_rows( s2_hat + m_hat^2,minibatch))
	  epsi_hat = exp(psi_hat)
	  phi_hat = epsi_hat/rowSums(epsi_hat)
	} else {
	  psi_hat = (y[idx]*m_hat) - (0.5*( s2_hat + m_hat^2))
	
	  epsi_hat = exp(psi_hat)
	  phi_hat = epsi_hat/sum(epsi_hat)
	}
	
	full_phi[idx,] = phi_hat
	weight = (step_counter	+ delay)^(-1.*forget)

	# Update variational distribution for mixture components
	natural_mu = (m_hat/s2_hat)
	natural_s2 = (-.5 / s2_hat)

	if (minibatch > 1){
	  phi_sum = colSums(phi_hat)/minibatch
	} else{
	  phi_sum = phi_hat
	}

	s2_intermediate = -.5*((tau^(-2)) + (N * phi_sum))
	s2_hat = ((1-weight)*natural_s2) + (weight*s2_intermediate)

	# reparamaterize s2_k
	s2_hat = -.5 / s2_hat

	if (minibatch > 1){
	  phi_prod = drop(crossprod(phi_hat, y[idx])) / minibatch
	}	else{
	  phi_prod = drop(phi_hat* y[idx])
	}

	m_hat_intermediate = N * phi_prod
  m_hat = ((1-weight)*natural_mu) + (weight*m_hat_intermediate)
  #m_hat = ((1-weight)*natural_mu) + (weight*m_hat_intermediate)

  # reparamaterize m_k
  m_hat = drop(m_hat*s2_hat)
  
	# Check ELBO (up to constant terms)
	E1 = {-1/(2*tau^2)}*sum(s2_hat + m_hat^2)
	#psi_hat = outer(y, m_hat) - (.5 /  ((tau^(-2) +1) )*rep_rows( s2_hat + m_hat^2, N))
	psi_hat = outer(y, m_hat) - 0.5*rep_rows(s2_hat + m_hat^2, N)
	E3 = sum(full_phi*psi_hat)
	E4 = sum(full_phi*log(full_phi))
	new_ELBO = E1 + E3 - E4
	ELBO_tracker[step_counter] = new_ELBO
	
	# check convergence
	delta = abs(new_ELBO - old_ELBO)
	converged = delta/(abs(old_ELBO) + rel_tol) < rel_tol
	if(!converged) {
		old_ELBO = new_ELBO
		step_counter = step_counter + 1
	}
}
ELBO_tracker = na.omit(ELBO_tracker)
converged; step_counter

plot(tail(ELBO_tracker, -10))

sort(m_hat)
sort(mu_true)

# Note: defining the predictive distribution is tricky.
# I'll cheat by calculating an average of the mixture indicators
# and use these as weights in a "posterior" mixture model
w_hat = colMeans(full_phi)

hist(y, 50, prob=TRUE)
curve(dnormix(x, mu_true, rep(1/K, K)), col='blue', lwd=2, add=TRUE)
curve(dnormix(x, m_hat, w_hat), col='red', lwd=2, add=TRUE)

# the naive unweighted predictive density
# won't work well if we get "collapsing" of mixture components
# in the variational approximation
curve(dnormix(x, m_hat, rep(1/K, K)), col='green', lwd=2, add=TRUE)
legend('topleft', lwd=2, col=c('blue', 'red', 'green'),
	legend=c("True", "Ad-hoc Weighted", "Naive Unweighted"))
	