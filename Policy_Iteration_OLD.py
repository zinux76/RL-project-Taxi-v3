import numpy as np
import gym
env = gym.make('Taxi-v3')

def policy_iteration(env):

    num_iterations = 10000 
    
    policy = np.zeros(env.observation_space.n)
    
    total_value_iterations = 0 

    for i in range(num_iterations):
        
        value_function, num_value_iteration = compute_value_function(policy) 
        print(" * Number of Value Iterations in Policy Iteration[%d] : %d"%(i+1, num_value_iteration)) 

        total_value_iterations += num_value_iteration 
        new_policy = extract_policy(value_function)

        if np.all(policy == new_policy):
            print("\nNumber of Policy Iterations required to converge :", i+1) 
            print("Number of Total Value Iterations :", total_value_iterations) 
            break

        policy = new_policy 

    return policy

def compute_value_function(policy, env=env): 

    num_iterations = 10000 
    threshold = 1e-20
    gamma = 0.99 

    value_table = np.zeros(env.observation_space.n) # 500 0's
        
    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)

        for s in range(env.observation_space.n): # 500 times
            
            a = policy[s]          
            branch = [] # empyt list for calculating weighted avg. in case of stochastic env.
            
            for prob, s_, r, _ in env.P[s][a]: # 1 time in case of deterministic env.     
                
                branch.append(prob * (r + gamma * updated_value_table[s_]))
                
            value_table[s] = sum(branch)

        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            break

    return value_table, i+1 # included "num_value_iteration" parameter

def extract_policy(value_table, env=env):

    gamma = 0.99
    policy = np.zeros(env.observation_space.n)
    
    for s in range(env.observation_space.n):
        
        Q_values = np.zeros(env.action_space.n)
            
        for a in range(env.action_space.n): 
                
            branch = []
                
            for prob, s_, r, _ in env.P[s][a]:         
                branch.append(prob*(r + gamma * value_table[s_]))
                
            Q_values[a] = sum(branch)        

        policy[s] = np.argmax(np.array(Q_values))
    
    return policy

def evaluate_policy(policy, env=env):
    num_episodes = 1000
    num_timesteps = 1000
    total_timestep = 0
    total_penalty = 0 # increase by 1 when executing “pickup” and “drop-off” actions illegally
    total_success = 0 

    for i in range(num_episodes):
        state = env.reset() 
        
        for t in range(num_timesteps):
            if policy is None:
                action = env.action_space.sample()
            else:
                action = policy[state]
                
            state, reward, done, info = env.step(action)
            
            if reward == -10: 
                total_penalty += 1 # increase by 1 when executing “pickup” and “drop-off” actions illegally

            if done:
                total_success += 1 
                break

        total_timestep += (t+1) 

    print(" * Number of successful episodes: %d / %d"%(total_success, num_episodes))
    print(" * Average number of penalties per episode: %.2f"%(total_penalty/num_episodes)) 
    print(" * Average number of timesteps per episode: %.2f"%(total_timestep/num_episodes))

print("\nFinding the Optimal Policy using Policy Iteration for Taxi-v3\n")
optimal_policy = policy_iteration(env)

print("\nFound the Optimal Policy for 500 states : ")
print(optimal_policy)

print("\nEvaluation using Random Policy")
evaluate_policy(None)

print("\nEvaluation using Optimal Policy")
evaluate_policy(optimal_policy)

print("\nIllustration of an Episode with Optimal Policy in Taxi game")
state = env.reset()
env.render()
done = False
while done != True:
    action = int(optimal_policy[state])
    state, reward, done, info = env.step(action)
    env.render()