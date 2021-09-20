"""
Wrappers for the OpenAI Gym environments:`SubProcVecEnv` and `DummyVecEnv`. They produce parallelized and serializedvectorized Gym environments for high-throughput training, respectively. To implement a custom environment, simply extend the OpenAI Gym Environment interface, register, and use with `ilpyt` as normal: https://github.com/openai/gym/blob/master/docs/creating-environments.md.
"""