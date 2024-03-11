# Learn JAX with Code

This repository offers examples and implementations to help understand some concepts and features in JAX. The features covered are more comprehensive in JAX, partly demonstrating why JAX is really great! Actual implementations in JAX are more complex than the simplified versions presented here, aimed at easier understanding. For a deeper comprehension, please refer to the [source code](https://github.com/google/jax) and [official documentation](https://jax.readthedocs.io/en/latest/).

## Understand Tracing

When we learn [JIT mechanics in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables), we know its key concepts:

* JIT and other JAX transforms operate by *tracing* a function, which involves analyzing how the function interacts with inputs of specific shape and type.

* **Static Variables** refer to those variables that you explicitly instruct JAX not to *trace*.

It works through a process known as "tracing", where `jax.jit` employs tracer objects to extract and record the sequence of operations specified by the function.

To deepen the understanding of "Tracing", let's simulate the tracing mechanism using NumPy in [tracer.py](https://github.com/yixiaoer/learn-jax-with-code/blob/main/tracer.py). This simulation will illustrate how a value is traced and under what circumstances it would not be traced.

The following printout showcases the trace of a test calculation. It does not trace specified static variables, recording their values directly instead. Other variables are assigned unique IDs. The sequence of operations is also logged in order. Below is the print result for the test calculation:

```shell
('reshape', v0, ((24, 2),), {'order': 'C'}, v1)
('__matmul__', v1, (array([[ 0.91908391, -1.16492133,  1.80642546, -0.40766693, -0.06169426,
         0.20667515,  0.15422442,  0.55837563, -0.24188108],
       [-0.35272848,  0.45473091, -1.14944122,  1.70440466, -0.85625506,
        -1.75825071, -0.50221464, -0.53137719, -0.94942683]]),), {}, v2)
('__add__', v2, (1.9826735513983742,), {}, v3)
('__add__', v3, (4.0,), {}, v4)
```

## Understand Automatic Differentiate

Automatic differentiation(autodiff or AD) is a computational technique for efficiently calculating partial derivatives of functions represented by computer programs. It is utilized in AI frameworks through forward and reverse modes.

JAX provides efficient implementations of both forward-mode autodiff (aka JVPs, Jacobian-Vector products) and reverse-mode autodiff (aka VJPs, Vector-Jacobian products), supporting higher-order derivatives as the derivative-computing functions are themselves differentiable, as further explored in [The Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#). It also enables advanced techniques, such as custom VJPs and JVPs, as shown in [Custom derivative rules](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html). The familiar `jax.grad` function is built on reverse-mode differentiation. Generally, reverse mode is more efficient in scenarios with more inputs than outputs, which is often the case in deep learning. 

To deepen the understanding of "Automatic Differentiate", let's implement reverse-mode autodiff using Numpy, detailed in [autodiff.py](https://github.com/yixiaoer/learn-jax-with-code/blob/main/autodiff.py).

To achieve autodiff via reverse mode, it is generally to construct a computational graph and apply the chain rule for backward calculation. Recording every transformation is essential, mirroring the approach used in the previously implemented `Tracer`. Additionally, basic operators have been again overloaded to integrate gradient calculation. Uncommenting the print statements within the function will display the partial derivatives of all input matrices with respect to the output scalar:

```shell
the partial derivative of input v0:
 [[-2.31498926 -0.15126296 -0.95344228 -0.45814965]
 [-2.31498926 -0.15126296 -0.95344228 -0.45814965]
 [-2.31498926 -0.15126296 -0.95344228 -0.45814965]
 [-2.31498926 -0.15126296 -0.95344228 -0.45814965]]
the partial derivative of input v1:
 [[ 0.0185569   0.03333231  0.0215656   0.00553173]
 [-0.18294742 -1.69888211 -1.7391486  -0.25367558]
 [ 0.03299103  0.03832882  0.01571867  0.03587593]
 [ 1.04148723  1.20911922  1.04072468 -0.35068197]]
the partial derivative of input v2:
 [[  7.15679709  -3.37794818  11.47929076 -13.34797813]
 [ -0.40382921   0.82943449  -1.81526981  -4.07413204]
 [ -3.82585338  -4.83445371  -0.82850604 -15.83623388]
 [-17.65459146  -4.78676227 -12.18075318  30.91683458]]
```
