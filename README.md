# Learn JAX with Code

A collection of code to help understand some features and implementations in JAX.

## Understand Automatic Differentiate

Automatic differentiation(autodiff or AD) is a computational technique for efficiently calculating partial derivatives of functions represented by computer programs. It is utilized in AI frameworks through forward and reverse modes.

JAX offers efficient and general implementations of both forward-mode and reverse-mode automatic differentiation, and the familiar `grad` function is built on reverse-mode, as further explored in [The Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.htm). Generally, reverse mode is more efficient in scenarios with more inputs than outputs, which is often the case in deep learning. 

To deepen the understanding of "Automatic Differentiate", let's implement reverse-mode autodiff using Numpy, detailed in [autodiff.py](https://github.com/yixiaoer/learn-jax-with-code/blob/main/autodiff.py)

## Understand Tracing

When we learn [JIT mechanics in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables), we know its key concepts:

* JIT and other JAX transforms operate by *tracing* a function, which involves analyzing how the function interacts with inputs of specific shape and type.

* **Static Variables** refer to those variables that you explicitly instruct JAX not to *trace*.

It works through a process known as "tracing", where `jax.jit` employs tracer objects to extract and record the sequence of operations specified by the function.

To deepen the understanding of "Tracing", let's simulate the tracing mechanism using NumPy in [tracer.py](https://github.com/yixiaoer/learn-jax-with-code/blob/main/tracer.py). This simulation will illustrate how a value is traced and under what circumstances it would not be traced.
