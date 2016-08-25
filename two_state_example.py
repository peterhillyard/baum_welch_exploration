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

# Set up the true Markov chain parameters and the true distribution parameters
pi_true = np.array([0.99,0.01])
A_true = np.array([[0.99,0.1],[0.5,0.5]])
state1_dist_params = np.array([0.,np.sqrt(1.)])
state2_dist_params = np.array([4.,np.sqrt(2.)])

# Now simulate the Markov chain so that we get the true states and a measurement
# for each state according to the distribtions
num_time_steps = 10
true_state_vec = np.zeros(num_time_steps)
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


