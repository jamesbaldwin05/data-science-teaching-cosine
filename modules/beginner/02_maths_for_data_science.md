# Probability Basics for Data Science

## Why probability?
Probability forms the foundation of data science, allowing us to model uncertainty, make predictions, and reason about data. Understanding probability helps you interpret models, evaluate risks, and make informed decisions in the presence of randomness.

---

## Table of Contents

1. [Basic Probability Concepts](#basic-probability-concepts)
2. [Conditional Probability](#conditional-probability)
3. [Bayes' Theorem](#bayes-theorem)
4. [Key Takeaways](#key-takeaways)
5. [Exercises](#exercises)
6. [Quiz](#quiz)

---

## Basic Probability Concepts

Probability measures how likely an event is to occur, expressed as a number between 0 (impossible) and 1 (certain).

- **Sample Space (S):** The set of all possible outcomes.
- **Event (A):** A subset of the sample space, i.e., one or more outcomes.

**Example:** For a fair coin toss:
- Sample space: S = {Heads, Tails}
- Event: A = {Heads}

### Rules of Probability

- **Addition Rule (Mutually Exclusive Events):**
  If A and B cannot both happen (mutually exclusive),  
  P(A or B) = P(A) + P(B)

- **Complement Rule:**
  The probability that event A does *not* happen:  
  P(not A) = 1 - P(A)

### Python Example: Probability Calculation

Let's estimate the probability of getting "Heads" in a fair coin toss using simulation.

```python
import random

trials = 10000
heads = 0

for _ in range(trials):
    if random.choice(['Heads', 'Tails']) == 'Heads':
        heads += 1

prob_heads = heads / trials
print(f"Estimated probability of Heads: {prob_heads}")
```

---

## Conditional Probability

Conditional probability is the probability of event A occurring given that event B has occurred.

- **Definition:**  
  P(A|B) = P(A and B) / P(B), where P(B) > 0

### Example

Suppose we have a deck of 52 cards.  
Let A = drawing an Ace, B = drawing a Spade.

- P(A) = 4/52
- P(B) = 13/52
- P(A and B) = 1/52 (the Ace of Spades)

So,
P(A|B) = P(A and B) / P(B) = (1/52) / (13/52) = 1/13

---

## Bayes' Theorem

Bayes' theorem allows us to reverse conditional probabilities, updating beliefs in light of new evidence.

- **Formula:**  
  P(A|B) = [P(B|A) * P(A)] / P(B)

### Example: Disease Testing

Suppose:
- 1% of people have a disease (P(Disease) = 0.01)
- Test is 99% accurate:  
  - P(Positive|Disease) = 0.99  
  - P(Positive|No Disease) = 0.01

If a person tests positive, what is the probability they actually have the disease?

#### Step 1: Compute P(Positive)
P(Positive) = P(Positive|Disease) * P(Disease) + P(Positive|No Disease) * P(No Disease)  
= (0.99 * 0.01) + (0.01 * 0.99)  
= 0.0099 + 0.0099  
= 0.0198

#### Step 2: Apply Bayes' Theorem
P(Disease|Positive) = [P(Positive|Disease) * P(Disease)] / P(Positive)  
= (0.99 * 0.01) / 0.0198  
≈ 0.5

### Python Example: Bayes' Theorem

```python
# Probabilities
P_D = 0.01          # Probability of disease
P_not_D = 0.99      # Probability of no disease
P_pos_given_D = 0.99
P_pos_given_not_D = 0.01

# Total probability of positive test
P_pos = P_pos_given_D * P_D + P_pos_given_not_D * P_not_D

# Bayes' theorem: probability of disease given positive test
P_D_given_pos = (P_pos_given_D * P_D) / P_pos

print(f"Probability of disease given positive test: {P_D_given_pos:.2f}")
```

---

## Key Takeaways

- Probability quantifies uncertainty and is fundamental in data science.
- The sample space includes all possible outcomes; events are subsets.
- Addition and complement rules help calculate probabilities.
- Conditional probability calculates the chance of one event given another.
- Bayes' theorem lets us update probabilities with new evidence.

---

## Exercises

1. **Simulate Coin Flips:**  
   Write Python code to simulate 1,000 coin flips. Estimate the probability of getting heads.

2. **Conditional Probability Calculation:**  
   Given 100 students: 40 like math, 30 like science, and 20 like both. What is the probability a randomly chosen student likes science, given they like math?

3. **Disease Test Posterior:**  
   A disease affects 2% of a population. A test is 95% accurate for both sick and healthy people. If a person tests positive, what is the probability they have the disease? (Show your calculation using Bayes' theorem.)

---

## Quiz

**Q1:** What is the sum of probabilities of all outcomes in a sample space?  
A) 1  
B) 0  
C) Depends  

**Q2:** What is the definition of conditional probability?  
A) Chance of two events at the same time  
B) Probability of one event given that another has occurred  
C) Probability of the opposite event  

**Q3:** What does Bayes’ theorem allow you to do?  
A) Calculate averages  
B) Reverse conditional probabilities to update beliefs  
C) Find maximum value