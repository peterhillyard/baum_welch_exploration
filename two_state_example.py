'''
Created on Aug 25, 2016

@author: pete
'''

# This script is used to simulate a markov chain with two states. In each state,
# we measure some kind of data (light, RSS, temperature, etc). In the simulation,
# we decide the true markov process parameters, and the true distribution 
# parameters. Then we pretend like we know nothing about the Markov chain. So in
# the beginning, we guess what the markov chain parameters are and the distributions
# for each state.
# 
# In the simulation, we use the true markov process to decide which state we are
# in and the measurement that we record. 
#
# We then run the measurements through the forward/backward algorithm and estimate 
# the state at each time. 
#
# We then run the baum-welch algorithm to estimate the mean and variance of the 
# distributions from each state. I want to see how well we can estimate these 
# parameters

import numpy as np

def get_gaus_prob(obs,dist_params):
    mu  = dist_params[0]
    var = dist_params[1]**2
    t1 = 1/np.sqrt(2*np.pi*var)
    t2 = (obs-mu)**2
    t3 = (2*var)
    return t1*np.exp(-t2/t3)
    
def my_round(val,decimal_place):
    return np.round(val*10**decimal_place)/10**decimal_place

# Set up the true Markov chain parameters and the true distribution parameters
pi_true = np.array([0.90,0.10])
A_true = np.array([[0.90,0.10],[0.10,0.90]])
state1_dist_params = np.array([0.0,np.sqrt(1.0)])
state2_dist_params = np.array([4.0,np.sqrt(2.0)])

# Set up the guesses for the distributions for each state and the HMM model 
# params
pi_est = np.array([0.90,0.10])
A_est = np.array([[0.90,0.10],[0.10,0.90]])
est_state1_dist_params = np.array([-0.5,np.sqrt(0.8)])
est_state2_dist_params = np.array([5.5,np.sqrt(2.3)])

# Now simulate the Markov chain so that we get the true states and a measurement
# for each state according to the distribtions
num_time_steps = 5000
true_state_vec = np.zeros(num_time_steps)
est_state_vec = np.zeros(num_time_steps)
est_state_vec2 = np.zeros(num_time_steps)
observation_vec = np.zeros(num_time_steps)

for ii in range(num_time_steps):
    
    rand_num = np.random.rand()
    
    if ii==0:
        # get initial state
        if rand_num <= pi_true[0]:
            cur_state = 1
        else:
            cur_state = 2
    else:
        if cur_state == 1:
            if rand_num <= A_true[0,0]:
                cur_state = 1
            else:
                cur_state = 2
        else:
            if rand_num <= A_true[1,0]:
                cur_state = 1
            else:
                cur_state = 2
    
    true_state_vec[ii] = cur_state
    
    rand_val = np.array([1,np.random.randn()])
    
    if cur_state == 1:
        observation_vec[ii] = np.sum(state1_dist_params*rand_val) 
    else:
        observation_vec[ii] = np.sum(state2_dist_params*rand_val)

# Now run the forward/backward algorithm to get gamma for each time step
iters = 100
for jj in range(iters):

    alpha = np.zeros((2,num_time_steps))
    beta = np.zeros((2,num_time_steps))
    gamma = np.zeros((2,num_time_steps))
    little_b = np.zeros((2,num_time_steps))
    
    for ii in range(num_time_steps):
        
        little_b[0,ii] = get_gaus_prob(observation_vec[ii],est_state1_dist_params)
        little_b[1,ii] = get_gaus_prob(observation_vec[ii],est_state2_dist_params)
        little_b[:,ii] = little_b[:,ii]/little_b[:,ii].sum()
        
        if ii == 0:
            alpha[:,ii] = little_b[:,ii]*pi_est
        else:
            alpha[:,ii] = np.dot(alpha[:,ii-1].T,A_est)*little_b[:,ii]
    
        alpha[:,ii] = alpha[:,ii]/np.sum(alpha[:,ii])
    
    for ii in range(num_time_steps-1,-1,-1):
        
        if ii == num_time_steps-1:
            beta[:,ii] = 1.
        else:
            beta[:,ii] = np.dot(A_est,beta[:,ii+1])*little_b[:,ii+1]
            beta[:,ii] = beta[:,ii]/np.sum(beta[:,ii])
            
        gamma[:,ii] = alpha[:,ii]*beta[:,ii]
        gamma[:,ii] = gamma[:,ii]/gamma[:,ii].sum()
        
        if gamma[0,ii] > gamma[1,ii]:
            est_state_vec[ii] = 1
        else:
            est_state_vec[ii] = 2
            
        if little_b[0,ii] > little_b[1,ii]:
            est_state_vec2[ii] = 1
        else:
            est_state_vec2[ii] = 2
                
    print np.sum(true_state_vec-est_state_vec==0)/float(num_time_steps)
    print np.sum(true_state_vec-est_state_vec2==0)/float(num_time_steps)
    print np.argwhere(np.abs(true_state_vec-est_state_vec)==1).size
        
#     est_mu1 = np.sum(gamma[0,:]*observation_vec)/np.sum(gamma[0,:])
#     est_mu2 = np.sum(gamma[1,:]*observation_vec)/np.sum(gamma[1,:])
#      
#     est_var1 = np.sum(gamma[0,:]*(observation_vec-est_mu1)**2)/(gamma[0,:].sum())
#     est_var2 = np.sum(gamma[1,:]*(observation_vec-est_mu2)**2)/(gamma[1,:].sum())
    
    est_mu1 = np.sum(little_b[0,:]*observation_vec)/np.sum(little_b[0,:])
    est_mu2 = np.sum(little_b[1,:]*observation_vec)/np.sum(little_b[1,:])
     
    est_var1 = np.sum(little_b[0,:]*(observation_vec-est_mu1)**2)/(little_b[0,:].sum())
    est_var2 = np.sum(little_b[1,:]*(observation_vec-est_mu2)**2)/(little_b[1,:].sum())
    
    est_state1_dist_params = np.array([est_mu1,np.sqrt(est_var1)])
    est_state2_dist_params = np.array([est_mu2,np.sqrt(est_var2)])
    
    print "---------------------------------------------------------"
    print "          | True Mean | Est. Mean | True Var | Est. Var |"
    print "---------------------------------------------------------"
    line1  = "| State 1 |    " + str(state1_dist_params[0]) + "    |   "
    line1 += str(my_round(est_mu1,3)) + "   |   "
    line1 += str(state1_dist_params[1]**2) + "    |   "
    line1 += str(my_round(est_var1, 3)) + "  |  "
    print line1
    print "---------------------------------------------------------"
    line1  = "| State 2 |    " + str(state2_dist_params[0]) + "    |   "
    line1 += str(my_round(est_mu2,3)) + "   |   "
    line1 += str(state2_dist_params[1]**2) + "    |   "
    line1 += str(my_round(est_var2, 3)) + "  |  "
    print line1
    print "---------------------------------------------------------\n"
