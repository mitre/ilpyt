"""
The runner coordinates the interaction between the agent and the environment. 
It collects transitions (state, action, reward, next state) over specified 
intervals of time. We can have the runner generate a collection of transitions 
for us by calling `generate_batch` (specify number of steps) and 
`generate_episodes` (specify number of episodes).
"""