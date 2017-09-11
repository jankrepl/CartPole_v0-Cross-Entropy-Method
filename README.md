# CartPole_v0 Cross Entropy Method
Solution to the CartPole_v0 environment using the general optimization technique **Cross Entropy Method**.

## Code

### Running
```
python Main.py
```

### Dependencies
*  gym
*  numpy

## Detailed Description
### Problem Statement and Environment
The goal is to move the cart to the left and right in a way that the pole on top of it does not fall down. The states 
of the environment are composed of 4 elements - **cart position** (x), **cart speed** (xdot),
**pole angle** (theta) and **pole angular velocity** (thetadot). For each time step when the pole is still on the cart
we get a reward of 1. The problem is considered to be solved if for 100 consecutive
episodes the average reward is at least 195.


If we translate this problem into reinforcement learning terminology:
* action space is **0** (left) and **1** (right)
* state space is a set of all 4-element lists with all possible combinations of values of x, xdot, theta, thetadot

---
### Cross entropy method
We are going to look for the optimal strategy without using any value functions - **actor only approach**.
We restrict ourselves to finding only deterministic policies and parametrize them in a following way:

```python
if w_0 * x + w_1 * xdot + w_2 * theta + w_3 * thetadot + b > 0: 
  return 1
else:
  return 0
```
Even though this parametrization is clearly restrictive, it will be sufficient to solve the simple CartPole_v0 environment

The Cross entropy method (**CEM**) is an optimization method that finds the
best policy (weights **w**). It can be described by the following pseudocode

```
set initial_parameters

while not found_winner: 
  sample policies from multivariate normal
  
  evaluate all policies
  
  select best performing policies
  
  reestimate mean vector and covariance matrix  
```
One important tweak is to **increase the variance** for each estimated covariance matrix. The goal of this is to
prevent the algorithm from being stuck in a suboptimal/local solution regions. In the code, we implement a **decaying 
noise coefficient**.

## Resources and links
* ![Learning Tetris Using the Noisy Cross-Entropy Method - Istvan Szita](http://iew3.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
