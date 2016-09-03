'''
Created on Aug 26, 2016

@author: pete
'''
import numpy as np

# This script simulates the situation where we have a multi-state and a multi-
# observation Markov model. So for example, let's say each state is a location
# in a building, and our N observations are from N sensors. We would like to 
# estimate the mean and variance of the multivariate gaussian distribution for
# each state.

def get_observation(mu,var):
    normal_rand = np.random.randn(mu.size)
    return np.sqrt(var)*normal_rand + mu

N = 3 # number of states
M = 10 # Number of observations

# Set up true Markov chain parameters and the distribution parameters
pi_true = np.ones(N)/N
A_true = np.ones((N,N))/N
mu_true = np.random.randint(0,10,size=(M,N))
var_true = np.random.randint(2,7,size=(M,N))/2.

# Set up our initial guess at the Markov chain and distribution parameters
pi_est = pi_true + 0*np.random.randn(N)
A_est  = A_true + 0*np.random.randn(N)
mu_est = mu_true + 1*np.random.randn(M,N)
var_est = var_true + 0.01*np.random.randn(M,N)

# Now simulate the Markov chain so that we get the true states and a measurement
# for each state according to the distribtions
T = 8000
true_state_vec = np.zeros(T)
est_state_vec = np.zeros(T)
est_state_vec2 = np.zeros(T)
est_state_vec3 = np.zeros(T)
observation_vec = np.zeros((M,T))

bounds = np.array([0] + np.cumsum(pi_est).tolist())
state_idx = np.arange(N)
cur_state = -1

for tt in range(T):
    
    rand_num = np.random.rand()
    
    if tt == 0:
        bounds = np.array([0] + np.cumsum(pi_est).tolist())
    else:
        bounds = np.array([0] + np.cumsum(A_est[cur_state,:]).tolist())
    
    cur_state = state_idx[((bounds[:-1] < rand_num) & (rand_num <= bounds[1:]))][0]
    
    observation_vec[:,tt] = get_observation(mu_est[:,cur_state],var_est[:,cur_state])
    true_state_vec[tt] = cur_state
        
# Now implement the forward/backward algorithm coupled with the baum-welch algorithm
iters = 10
for jj in range(iters):
    alpha = np.zeros((N,T))
    beta = np.zeros((N,T))
    gamma = np.zeros((N,T))
    little_b = np.zeros((N,T))
    
    for tt in range(T):
        
        # Compute likelihood
        for ii in range(N):
            t1 = ((2*np.pi)**(M/2.))*(np.prod(var_est[:,ii])**0.5)
            t2 = np.sum(-((observation_vec[:,tt]-mu_est[:,ii])**2)/(2*var_est[:,ii]))
            little_b[ii,tt] = np.exp(t2)/t1
        little_b[:,tt] = little_b[:,tt]/little_b[:,tt].sum()
        
        # Compute forward terms
        if tt == 0:
                alpha[:,tt] = little_b[:,tt]*pi_est
        else:
            alpha[:,tt] = np.dot(alpha[:,tt-1].T,A_est)*little_b[:,tt]
        alpha[:,tt] = alpha[:,tt]/np.sum(alpha[:,tt])
    
    for tt in range(T-1,-1,-1):
        
        # Compute beta
        if tt == T-1:
            beta[:,tt] = 1.
        else:
            beta[:,tt] = np.dot(A_est,beta[:,tt+1])*little_b[:,tt+1]
            beta[:,tt] = beta[:,tt]/np.sum(beta[:,tt])
        
        # Compute gamma
        gamma[:,tt] = alpha[:,tt]*beta[:,tt]
        gamma[:,tt] = gamma[:,tt]/gamma[:,tt].sum()
        
        # Get current state estimate
        est_state_vec[tt] = state_idx[gamma[:,tt] == gamma[:,tt].max()][0]
        est_state_vec2[tt] = state_idx[little_b[:,tt] == little_b[:,tt].max()][0]
        est_state_vec3[tt] = state_idx[alpha[:,tt] == alpha[:,tt].max()][0]
        
    print np.sum(true_state_vec-est_state_vec==0)/float(T)
    print np.sum(true_state_vec-est_state_vec2==0)/float(T)
    print np.sum(true_state_vec-est_state_vec3==0)/float(T)
    
    tmp_mu = np.zeros((M,N))
    tmp_var = np.zeros((M,N))
    for ii in range(N):
#         gamma_mat = np.tile(gamma[ii,:],(M,1))
#         tmp_mu[:,ii]  = np.sum(gamma_mat*observation_vec,axis=1)/gamma[ii,:].sum()
#         tmp_var[:,ii] = np.sum(gamma_mat*(np.tile(tmp_mu[:,ii],(T,1)).T-observation_vec)**2,axis=1)/gamma[ii,:].sum()
        
        little_b_mat = np.tile(little_b[ii,:],(M,1))
        tmp_mu[:,ii]  = np.sum(little_b_mat*observation_vec,axis=1)/little_b[ii,:].sum()
        tmp_var[:,ii] = np.sum(little_b_mat*(np.tile(tmp_mu[:,ii],(T,1)).T-observation_vec)**2,axis=1)/little_b[ii,:].sum()
        
        mu_est = tmp_mu.copy()
        var_est = tmp_var.copy()
        
#     print tmp_mu
#     print mu_true
#     print tmp_var
#     print var_true