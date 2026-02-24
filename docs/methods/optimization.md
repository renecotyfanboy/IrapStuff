---
icon: fontawesome/solid/diagram-project
---

# Optimization

"*Optimization consists of maximizing or minimizing a real function by systematically choosing input values from within an allowed set and computing the value of the function.*" [Wikipedia](https://en.wikipedia.org/wiki/Mathematical_optimization)

## A (very) brieve introduction 

Optimization is a very broad topic, and is often about finding the best input values for a given function. There is 
a tremendous number of situations where you might want to find \(\hat{\theta}\) such that 

\[\hat{\theta} = \underset{\theta \in \mathbb{R}^n}{\text{argmin}} \, f(\theta)\]

where \(f\) is some n-dimensional function that we refer to as the objective function, or cost function. 

!!! note 
    Any minimization problem can be reformulated as a maximization problem and vice versa with a sign change as for most of the situations we can write \(\hat{\theta} = \underset{\theta \in \mathbb{R}^n}{\text{argmin}} \, f(\theta) = \underset{\theta \in \mathbb{R}^n}{\text{argmax}} \, -f(\theta)\)