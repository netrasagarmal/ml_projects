## Logistic Regression?

Despite the name, **logistic regression is a classification algorithm**. It's used when the **target variable is binary**, i.e., has two possible values:

* Example: Spam (1) or Not Spam (0), Pass (1) or Fail (0)

It models the **probability** that a given input belongs to class 1 (positive class), using the **logistic (sigmoid) function**.

---

## Goal of Logistic Regression

We want to build a model that can **predict the probability** that a given input $x$ belongs to class 1 (positive class), where the output $y \in \{0, 1\}$.

---

## 1. **Linear Combination (Same as Linear Regression)**

We start by calculating a **linear score**:

$$
z = w^\top x + b = \sum_{i=1}^{n} w_i x_i + b
$$

Where:

* $x \in \mathbb{R}^n$ is the input feature vector
* $w \in \mathbb{R}^n$ is the weight vector (parameters)
* $b$ is the bias
* $z$ is a **real-valued number** (unbounded)

But we want to model a **probability**, so this value needs to be **squashed** between 0 and 1.

---

## 2. **Sigmoid Function: Mapping Real Numbers to \[0, 1]**

We apply the **sigmoid function** $\sigma(z)$ to squash the output:

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

This output $\hat{y} \in (0, 1)$ can now be interpreted as a **probability**:

> "The probability that the output is class 1, given input x"

The sigmoid has useful properties:

* It’s **differentiable**
* Output is **bounded** between 0 and 1
* It has a **nice gradient** (we’ll use that later)

---

## 3. Why Not Use Linear Regression for Classification?

In linear regression:

$$
\hat{y} = w^\top x + b
$$

* The output is unbounded (can be <0 or >1)
* No probabilistic interpretation
* MSE loss isn’t well-suited for classification

**In contrast, logistic regression:**

* Produces probabilities
* Uses a loss function derived from likelihood theory

---

## 4. **Likelihood and Cross-Entropy Loss**

### Step 1: Model the data probabilistically

We assume:

$$
P(y = 1 \mid x) = \hat{y} = \sigma(w^\top x + b) \\
P(y = 0 \mid x) = 1 - \hat{y}
$$

We can combine this using the true label $y \in \{0, 1\}$ as:

$$
P(y \mid x) = \hat{y}^y (1 - \hat{y})^{1 - y}
$$

---

### Step 2: Maximum Likelihood Estimation (MLE)

We want to **maximize the likelihood** over all training samples:

$$
L(w, b) = \prod_{i=1}^{n} \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}
$$

This is inconvenient to optimize due to the product. So we take the **log-likelihood**:

$$
\log L(w, b) = \sum_{i=1}^{n} y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)
$$

This is called the **log-loss** or **binary cross-entropy**.

---

### Step 3: Define Cost Function (Loss to Minimize)

Instead of maximizing likelihood, we minimize **negative log-likelihood**:

$$
J(w, b) = - \frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

Why is this a good choice?

* Convex in parameters (good for optimization)
* Harshly penalizes confident wrong predictions
* Probabilistically sound

---

## 5. **Gradient Descent for Optimization**

We now minimize the cost function using **gradient descent**.

---

### Derivatives:

Let’s compute gradients for a single training sample:

1. Prediction:

$$
\hat{y}_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}, \quad z_i = w^\top x_i + b
$$

2. Gradient of cost w\.r.t. weights:

$$
\frac{\partial J}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) x_i
$$

3. Gradient w\.r.t. bias:

$$
\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
$$

---

### Parameter Update Rule

At each step:

$$
w := w - \alpha \cdot \frac{\partial J}{\partial w} \\
b := b - \alpha \cdot \frac{\partial J}{\partial b}
$$

Where:

* $\alpha$ is the **learning rate**

We repeat this until convergence.

---

## Summary

| Component          | Purpose                                 |
| ------------------ | --------------------------------------- |
| $z = w^\top x + b$ | Raw score before activation             |
| $\sigma(z)$        | Converts raw score into probability     |
| Log-Loss           | Measures how good the prediction is     |
| Gradients          | Tell us how to improve the parameters   |
| Gradient Descent   | Optimization algorithm to minimize loss |

---

## Final Prediction Rule

Once trained, you predict using:

$$
\hat{y} = \sigma(w^\top x + b)
$$

* If $\hat{y} \geq 0.5$, predict class 1
* Else, predict class 0

