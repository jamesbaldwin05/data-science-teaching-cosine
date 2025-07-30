# Bayesian Inference

**Concept**  
Bayesian inference updates probability based on evidence using Bayes' rule. It is fundamental for probabilistic modeling. For simple cases, you can compute by hand; for complex, use libraries like pymc.

### Example
```python
# Probability of having a disease given a positive test (Bayes theorem)
# P(D|T) = P(T|D)*P(D) / [P(T|D)*P(D) + P(T|~D)*P(~D)]
p_disease = 0.01
p_pos_given_disease = 0.99
p_pos_given_no_disease = 0.05

p_no_disease = 1 - p_disease
numer = p_pos_given_disease * p_disease
denom = numer + p_pos_given_no_disease * p_no_disease
posterior = numer / denom
print(f"Probability of disease given positive test: {posterior:.2%}")
```

### Exercise
"""
If the disease prevalence drops to 0.001, what is the new probability of disease given a positive test?
"""
```python
# Modify the example above with p_disease = 0.001
# Your code here
```

### Quiz
**Q1:** What is the core equation for Bayesian inference?
- A) Law of Large Numbers
- B) Bayes' rule
- C) Central Limit Theorem
- D) Markov property
**A:** B

**Q2:** In Bayes' theorem, what does P(D|T) represent?
**A:** probability of disease given a positive test