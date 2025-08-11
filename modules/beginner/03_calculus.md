# Calculus
Calculus is the branch of mathematics that deals with continuous change through derivatives and integrals.
In machine learning, many algorithms focus on optimization—adjusting model parameters to minimize a loss (or error) function.
Calculus gives us the tools to measure how small changes in these parameters affect the loss, helping us update them efficiently to improve model performance.

---

## Table of Contents
1. [Differentiation](#differentation)
2. [Partial Differentiation](#partial-differentiation)
3. [Gradient Vectors](#gradient-vectors)
4. [Integration](#integration)

---

### Differentation

#### Derivatives 
- Measure instantaneous rate of change of a function at a point. For a function $f(x)$, the derivative $f'(x)$ tells us how $f(x)$ changes at $x$.
- It is defined by a limit:  
$f'(x) = \displaystyle\lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$  

- Some common derivatives which can be used instead of using the limit definition are:  

| Function Name          | Function $f(x)$              | Derivative $f'(x)$                 | Example Function            | Example Derivative           |
|-----------------------|-----------------------------|----------------------------------|----------------------------|-----------------------------|
| Constant              | $c$                         | $0$                              | $5$                        | $0$                         |
| Polynomials           | $x^n$                      | $n x^{n-1}$                     | $x^3$                      | $3x^2$                      |
| Constant multiple     | $c \cdot f(x)$             | $c \cdot f'(x)$                 | $4x^3$                     | $4 \cdot 3x^2 = 12x^2$      |
| Sum / Difference      | $f \pm g$                  | $f' \pm g'$                     | $x^3 + x^2$                | $3x^2 + 2x$                 |
| Exponential natural   | $e^x$                      | $e^x$                          | $3e^x$                     | $3e^x$                      |
| Exponential base a    | $a^x$                      | $a^x \ln(a)$                   | $2^x$                      | $2^x \ln(2)$                |
| Natural logarithm     | $\ln(x)$                   | $\frac{1}{x}$                  | $3\ln(x)$                   | $\frac{3}{x}$               |
| Logarithm base a      | $\log_a(x)$                | $\frac{1}{x \ln(a)}$           | $\log_2(x)$                | $\frac{1}{x \ln(2)}$        |
| Sine                  | $\sin{x}$                  | $\cos{x}$                     | $4\sin{x}$                 | $4\cos{x}$               |
| Cosine                | $\cos{x}$                  | $-\sin{x}$                    | $6\cos{x}$                | $ -6\sin{x}$       |

- Functions such as $\frac{1}{x}$ can be differentiated by rewritting them as polynomials ($\frac{1}{x} = x^{-1}$) and using the above rule.

- For example, for $f(x) = 3x^4 + \dfrac{1}{x^2} + 2e^x - ln(x) - 2sin(x),\;$ the derivative is $f'(x) = 12x^3 - \dfrac{2}{x^3}+ 2e^x - \dfrac{1}{x} -2 cos(x)$.  

- Note there are different notations for derivatives, if a function is declared as $y= 3x^4 + 2e^x - ln(x),\;$ the derivative is commonly written as $\frac{dy}{dx} = 12x^3 + 2e^x - \dfrac{1}{x}$ and this means the same thing as $f'(x)$

- We can take derivatives with respect to any variable, if we had a function $x = t^2 + 2t + 1$, the derivative with respect to t, $\frac{dx}{dt} = 2t + 2$.

#### Chain Rule
- The chain rule can be used to differentiate more complicated functions. It is defined as follows:  
For $y=f(x)$ and $z=g(y)$,  
$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$  

- This can be thought of as the "$dy$" cancelling but is nothing more than this (derivatives are not fractions). It is commonly used when we have a function 
within a function.  

- For example:  
$z=(3x+2)^5$  
$let \; y=3x+2\; => z=y^5$  
$\frac{dy}{dx} = 3$ and $\frac{dz}{dy} = 5y^4$  
$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$  
${\frac{dz}{dx}} = 3 \cdot 5y^4$  
${\frac{dz}{dx}} = 15(3x+2)^4$

- Instead of having to declare a new variable every time, many cases of the chain rule can be thought of as multiplying the derivative of the embedded function with the derivative of the outside function.  

- For example:
$f(x) = 4\sin(3x)$  
The derivative of the inside function is $3$ so $f'(x) = 4 \cdot 3 \cdot cos(3x)$ = $12\cos(3x)$

- A more complicated example:
$f(x) = 4\cos((2x+1)^3)$  
$f'(x) = 4 \cdot 2 \cdot 3(2x+1)^2 \cdot -\sin((2x+1)^3) = -24(2x+1)^2\sin((2x+1)^3)$

#### Higher-Order Derivatives
- We can take the derivative multiple times to give us further expressions how the rate of change itself changes.
- A higher-order derivative is found by taking the derivative of a derivative.
-They are written as $f''(x)$ or $\frac{d^{\;2}y}{dx^{\;2}}$ for the second derivative, $f'''(x)$ or $\frac{d^{\;3}y}{dx^{\;3}}$ for the third derivative and $f^{(n)}(x)$ or $\frac{d^{\;n}y}{dx^{\;n}}$ for the n-th derivative (once it becomes impractical to use dashes next to the function).  
- For example,  
$f(x) = 3x^4 + 2e^x,\;$  
$f'(x) = 12x^3 + 2e^x - \dfrac{1}{x}$  
$f''(x) = 36x^2 + 2e^x$  
$f'''(x) = 72x + 2e^x$  
$f^{(4)}(x) = 72 + 2e^x$  
$f^{(5)}(x) = 2e^x$ (=$\frac{d^{\;5}y}{dx^{\;5}}$ if $y = f(x)$)

### Partial Differentiation

#### Partial Derivatives
- If a function has multiple variables, we cannot take the derivative in the same way as with single-variable functions, since an ordinary derivative measures the rate of change with respect to a single variable.  
- Instead, we use partial derivatives, which measure the rate of change of the function with respect to one variable while keeping the others constant.
- Partial derivatives involve treating every variable other than one as a constant and then taking derivatives in the normal way.
- They are written as $f_x$ or $\frac{\partial f}{\partial x}$.  
- For example, if $f(x,y) = x^2y + (3y+2)^2,$  
$f_x = 2xy$ (the partial derivative of the function with respect to $x$, treating $y$ as a constant) and $f_y = x^2 + 6(3y+2)$ (the partial derivative of the function with respect to $y$, treating $x$ as a constant).  
- Partial differentiation can be applied to a function of any number of variables by treating all but one as constant.

#### Higher-Order Partial Derivatives
- We can take higher order derivatives of a multivariable function in a similar way, commonly written with the number of derivatives as a subscript of the function such as $f_{xx}$ for the second order partial derivative with respect to x (other notation explained later).
- For example,  
$f(x,y) = x^2y + 3y^2$  
$f_x = 2xy$  
$f_y = x^2 + 6y$  
$f_{xx} = 2y$  
$f_{yy} = 6$.  

#### Mixed Partial Derivatives
- We can also find mixed derivatives, which are derivatives which involve differentiating with respect to different variables in a sequence.
- If we were to derive a function first with respect to $x$, then with respect to $y$, we would write it as $f_{xy}$.
- We read from left to right to see the order of the variables the function was derived with respect to.
- For example,  
$f(x,y) = x^2y + 3y^2$  
$f_x = 2xy$  
$f_{xy} = 2x$ (by differeniating with respect to $x$ then $y$).  
$g(x,y) = x^3y^2 + 4xy$  
$g_x = 3x^2y^2 + 4y$  
$g_{xy} = 6x^2y + 4$  
$g_{xyx} = 12xy$  
$g_{xyxx} = 12y$ (by differentiating with respect to $x, y, x, x$ in that order).  

- **Symmetry of Mixed Partial Derivatives**: In general, any partial derivatives that involve the same total number of differentiations with respect to each variable will be equal, regardless of the order in which they are taken. This means that $f_{xy} = f_{yx}$ or even $f_{xyyx} = f_{xxyy}$ (since we have differentiated with respect to both variables twice).

- **Other Notation for Mixed Partial Derivatives**: The notation now gets confusing since the other way of writing mixed partial derivatives, such as $\frac{\partial^2 f}{\partial x \partial y}$, reads as the opposite: we read the order they have been differentiated in from right to left. For example,  
$f(x,y) = x^3y^2 + 4xy$  
$\frac{\partial f}{\partial y} = 2x^3y + 4x \;\\[4pt]
\frac{\partial^2 f}{\partial x \partial y} = 6x^2y + 4 \; =\frac{\partial}{\partial x}(\frac{\partial f}{\partial y})\;\\[4pt]
\frac{\partial^3 f}{\partial x^2 \partial y} = 12xy \; =\frac{\partial}{\partial x}(\frac{\partial^2 f}{\partial x \partial y}) \;$ ($=f_{yxx}$ in other notation, notice the difference in order).

### Gradient Vectors
- The gradient vector of a function $f(x_1, x_2, ..., x_n)$ is a vector that points in the direction of the greatest rate of increase of the function.
- It is made up of all the partial derivatives of the function with respect to each variable.
- For a function $f$, the gradient vector $\nabla f$ is:  
$\nabla f = (\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}) = f(f_{x_1}, f_{x_2},...,f_{x_n})$
- For example, to find the gradient vector of the point $(1, 2, -1)$ on the function $f(x,y,z) = x^2y+3yz^2-e^{xz}$:  
$f_x = 2xy-ze^{xz}$  
$f_z= x^2+3z^2$  
$f_y= 6yz-xe^{xz}$  
$\nabla f = (f_x, f_y, f_z) = (2xy-ze^{xz}, x^2+3z^2, 6yz-xe^{xz})$  
Substitute $x=1, y=2, z=-1$ to find $\nabla f = (4+\frac{1}{e},\; 4,\; -12-\frac{1}{e})$.

- Geographically, this means that at the point $(1,2,-1)$ the function is increasing fastest in the direction of the vector $(4+\frac{1}{e},\; 4,\; -12-\frac{1}{e})$.

- You can find the length of this vector to tell you how steep the slope is as this point.

### Integration

#### Integrals
- Integration is the opposite of differentiation. 

### Calculus in Optimization Algorithms
- Gradient Descent
  - Iteratively updates parameters by moving opposite the gradient to reduce the loss.
  - Core idea: take small steps in the direction that locally reduces error most.
- Stochastic Gradient Descent (SGD)
  - Uses subsets (mini-batches) of data to estimate gradients.
  - Faster, noisier updates that scale to large datasets and can help escape shallow local minima.
- Convergence
  - Learning rate (step size) and gradient magnitudes determine stability and speed.
  - Too large: divergence/oscillation; too small: slow learning. Adaptive methods adjust step sizes automatically.

#### You should be able to:
- Explain derivative, partial derivative, gradient, and chain rule in ML terms.
- Interpret the gradient’s direction and why updates move against it in minimization.
- Describe gradient descent vs. SGD and the role of the learning rate in convergence.
- Connect integrals to expectations and marginalization in probabilistic models.

---