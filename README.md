## Comparing Value Iteration and Policy Iteration in "Taxi-v3" environment

The background of this project was to demonstrate the following two facts in [this document](https://www.baeldung.com/cs/ml-value-iteration-vs-policy-iteration).
 * Value iteration and policy iteration algorithms are both guaranteed to converge to an optimal policy in the end.
 * Yet, the policy iteration algorithm converges within fewer iterations. 
 
Refer to [Gym Documentation](https://www.gymlibrary.dev/environments/toy_text/taxi/) for "Taxi-v3" environment.

Running <Value_Iteration.py> and <Policy_Iteration_NEW.py> will show the following results in common.
1. Number of iterations required to converge to the optimal policy
2. Optimal policy derived through reinforcement learning :   
   - Choosing an action out of 6 possible actions for 500 states in this "Taxi-v3" environment
3. Evaluation using random policy and optimal policy in both algorithms respectively :  
   - Number of successful episodes
   - Average number of penalties per episode
   - Average number of timesteps per episode
4. Illustration of an episode with optimal policy in Taxi-game

You can check the following facts from the results.
1. The optimal policies derived from two algorithms are the same.
2. Optimal policy has improved performance significantly than using random policy in both algorithms.
3. Trained taxi picks up and drops off passengers in the optimal path.

Regarding the number of iterations required to converge to the optimal policy :     
In order to accurately calculate the number of total iterations, I should have created a GLOBAL_COUNTER variable and exchanged it as a parameter in each module.
In the next update, I will upload the modified code.
