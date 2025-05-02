## Simple Linear Regression
**Simple Linear Regression** is a statistical method used to model the relationship between two variables:

* One **independent variable** (input, usually denoted as `x`)
* One **dependent variable** (output, usually denoted as `y`)

It fits a **straight line** (linear equation) to the data that best predicts the value of `y` based on `x`.

---

#### The Linear Equation

The model is defined as:

$$
\hat{y} = mx + b
$$

Where:

* $\hat{y}$ is the **predicted value**
* $x$ is the input variable
* $m$ is the **slope** of the line (how much y changes per unit change in x)
* $b$ is the **intercept** (value of y when x = 0)

---

#### Purpose

Simple linear regression is used to:

* **Predict outcomes** based on input
* **Understand the strength and direction** of the relationship between two variables
* **Quantify** the influence of one variable on another

---

#### Example

If we want to predict a student’s exam score based on hours studied:

* `x = hours studied`
* `y = exam score`

Simple linear regression would find the **best straight line** through the data points to model this relationship.


### Goal of Simple Linear Regression

We are trying to **model the relationship** between:

* One **independent variable** $x$ (e.g., hours studied)
* One **dependent variable** $y$ (e.g., exam score)

We assume that the relationship between them is **linear**, so we want to fit a straight line:

$$
\hat{y} = mx + b
$$

Where:

* $\hat{y}$ is the **predicted value**
* $m$ is the **slope** (how much y changes for each unit change in x)
* $b$ is the **intercept** (value of y when x = 0)

---

### Step 1: The Core Idea — Fit a Line That Minimizes Error

We want to choose the values of `m` and `b` such that the predicted values $\hat{y}$ are as close as possible to the actual values $y$.

So the **core problem** becomes:

> Find `m` and `b` that minimize the total prediction error.

---

### Step 2: Measuring "How Wrong" We Are — Loss Function

To do this, we use a **loss function**. The most common choice is **Mean Squared Error (MSE)**:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Why MSE?

* It penalizes large errors more than small ones (because of squaring)
* It's smooth and differentiable (so we can optimize it easily)
* It gives a clear geometric interpretation: average squared vertical distance from the actual points to the line

So:

> Lower MSE means better line fit.

---

### 🔧 Step 3: Optimizing the Line — Using Calculus

We want to **minimize the MSE**. There are two ways:

#### 1. **Analytical method** (closed-form solution):

We use calculus to derive formulas for the best `m` and `b` directly. This works for small/simple problems.

#### 2. **Gradient Descent** (iterative optimization):

This is more flexible and generalizable (especially when data or models get complex). We take steps in the direction that reduces the MSE.

---

### Step 4: Gradient Descent — How It Works

#### Key concept:

We treat the MSE as a **surface** over parameters `m` and `b`. Gradient descent is like a ball rolling downhill toward the lowest point (minimum loss).

At each step:

1. Compute the **gradient** (partial derivatives) of MSE w\.r.t. `m` and `b`:

   * $\frac{\partial \text{MSE}}{\partial m} = -\frac{2}{n} \sum x_i(y_i - \hat{y}_i)$
   * $\frac{\partial \text{MSE}}{\partial b} = -\frac{2}{n} \sum (y_i - \hat{y}_i)$

2. Update parameters:

   * $m = m - \alpha \cdot \frac{\partial \text{MSE}}{\partial m}$
   * $b = b - \alpha \cdot \frac{\partial \text{MSE}}{\partial b}$

   Here, $\alpha$ is the **learning rate** — it controls how big the steps are.

3. Repeat until the updates get very small (converge).

---

### 🔁 Step 5: Model Converges to Best-Fit Line

After many iterations, gradient descent finds values for `m` and `b` that **minimize the MSE** — in other words, the **best-fit line** through the data.

Now, this line can be used to **predict** values of `y` for new inputs `x`.

---

### Summary

| Concept             | Intuition                                                                |
| ------------------- | ------------------------------------------------------------------------ |
| Linear Model        | Predicts y as a straight line: $\hat{y} = mx + b$                        |
| Loss Function (MSE) | Measures how far predictions are from actual data                        |
| Optimization Goal   | Find `m` and `b` that minimize the MSE                                   |
| Gradient Descent    | An iterative way to reduce the error by adjusting `m` and `b`            |
| Learning Rate (α)   | Controls how big each update step is                                     |
| Convergence         | When the model reaches the lowest possible loss — best-fit line is found |


