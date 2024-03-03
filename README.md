# Learn JAX with Code

A collection of code to help understand some features and implementations in JAX.

## Understand Tracing

When we learn [JIT mechanics in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables), we know its key concepts:
* JIT and other JAX transforms operate by *tracing* a function, which involves analyzing how the function interacts with inputs of specific shape and type.

* **Static Variables** refer to those variables that you explicitly instruct JAX not to *trace*.

It works through a process known as "tracing", where `jax.jit` employs tracer objects to extract and record the sequence of operations specified by the function.

To deepen the understanding of "Tracing", let's simulate the tracing mechanism using NumPy in [tracer.py](https://github.com/yixiaoer/learn-jax-with-code/blob/main/tracer.py). This simulation will illustrate how a value is traced and under what circumstances it would not be traced.
