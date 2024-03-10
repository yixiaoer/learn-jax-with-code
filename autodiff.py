import jax; jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
jax.config.update("jax_enable_x64", True)

from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from tracer import Context, IDGenerator, Tracer

class GradArray(Tracer):
    def __init__(self, arr: np.ndarray, ctx: Context, id_: str | None = None, records: list[tuple] | None = None):
        super().__init__(arr, ctx, id_, records)

    def __add__(self, other: Any):
        arr, ctx, records, id_ = self.arr, self.ctx, self.records, self.id
        is_float = isinstance(other, float)
        is_GradArray = isinstance(other, GradArray)
        if is_float or is_GradArray:
            id_new = ctx.id_gen()
            other_ = other if is_float else other.id
            add_res = arr + other if is_float else arr + other.arr

            def grad_operator(grad: np.ndarray):
                if is_GradArray:
                    return grad, grad
                return (grad,)

            if records is not None:
                records.append(('__add__', id_, (other_,), {}, id_new, grad_operator))
            return GradArray(add_res, ctx, id_=id_new, records=records)
        raise NotImplementedError('Not supported.')

    def __matmul__(self, other: Any):
        arr, ctx, records, id_ = self.arr, self.ctx, self.records, self.id
        id_new = ctx.id_gen()
        other_ = other.id
        mat_res = arr @ other.arr

        def grad_operator(grad: np.ndarray):
            g_self = grad @ other.arr.T
            g_other = self.arr.T @ grad
            return g_self, g_other

        if records is not None:
            records.append(('__matmul__', id_, (other_,), {}, id_new, grad_operator))
        return GradArray(mat_res, ctx, id_=id_new, records=records)

    def sum(self, *args, **kwargs):
        arr, ctx, records, id_ = self.arr, self.ctx, self.records, self.id
        id_new = ctx.id_gen()

        def grad_operator(grad: np.ndarray):
            g_self = np.ones(arr.shape) * grad
            return (g_self,)

        if records is not None:
            records.append(('sum', id_, args, kwargs, id_new, grad_operator))
        return GradArray(np.array(arr.sum()), ctx, id_=id_new, records=records)

    def exp(self, *args, **kwargs):
        arr, ctx, records, id_ = self.arr, self.ctx, self.records, self.id
        id_new = ctx.id_gen()

        def grad_operator(grad: np.ndarray):
            g_self = np.exp(arr) * grad
            return (g_self,)

        if records is not None:
            records.append(('exp', id_, args, kwargs, id_new, grad_operator))
        return GradArray(np.exp(arr), ctx, id_=id_new, records=records)

    def sin(self, *args, **kwargs):
        arr, ctx, records, id_ = self.arr, self.ctx, self.records, self.id
        id_new = ctx.id_gen()

        def grad_operator(grad: np.ndarray):
            g_self = np.cos(arr) * grad
            return (g_self,)

        if records is not None:
            records.append(('sin', id_, args, kwargs, id_new, grad_operator))
        return GradArray(np.sin(arr), ctx, id_=id_new, records=records)

def grad_and_value(f: Callable):
    def h(*args, **kwargs):
        id_gen = IDGenerator()
        variables = {}
        ctx = Context(id_gen, variables)

        records = []
        args = [GradArray(arg, ctx, records=records) for arg in args]
        kwargs = {k: GradArray(v, ctx, records=records) for k, v in kwargs.items()}
        inputs_vid = [f'v{i}' for i in range(len(args) + len(kwargs))]
        out = f(*args, **kwargs)
        grads = {}
        grads[str(out.id)] = np.array(1.)
        for op, obj, args, kwargs, id_new, grad_operator in reversed(records):
            grad_op_out = grads[str(id_new)]
            grad_op_inputs = grad_operator(grad_op_out)
            inputs = obj, *args, *kwargs.keys()
            for input_, grad_op_input in zip(inputs, grad_op_inputs):
                if input_ not in grads:
                    grads[str(input_)] = grad_op_input
                else:
                    grads[str(input_)] += grad_op_input

        return out.arr, [grads[arg] for arg in inputs_vid][0]
    return h

def grad(f: Callable):
    def h(*args, **kwargs):
        _, grad = grad_and_value(f)(*args, **kwargs)
        return grad
    return h

def main():
    # test calculation function
    def f(a,b,c):
        if isinstance(a, np.ndarray):
            return (a @ np.sin(b) + np.exp(b @ c)).sum() + 5.
        return (a @ GradArray.sin(b) + GradArray.exp(b @ c)).sum() + 5.

    np.random.seed(42)

    a = np.random.randn(4, 4)
    b = np.random.randn(4, 4)
    c = np.random.randn(4, 4)
    f_grad_value = grad_and_value(f)
    out, grad_a = f_grad_value(a, b, c)

    # check with JAX
    a_jax = jnp.asarray(a, dtype=jnp.float64)
    b_jax = jnp.asarray(b, dtype=jnp.float64)
    c_jax = jnp.asarray(c, dtype=jnp.float64)

    def f_jax(a_jax, b_jax,c_jax):
        return jnp.sum(jnp.dot(a_jax,np.sin(b_jax)) + jnp.exp(jnp.dot(b_jax, c_jax))) + 5.

    out_jnp = f_jax(a_jax, b_jax, c_jax)
    grad_a_jax = jax.grad(f_jax)(a_jax, b_jax, c_jax)

    print(jnp.allclose(out, out_jnp))  # True
    print(jnp.allclose(jnp.asarray(grad_a, dtype=jnp.float64), grad_a_jax))  # True

if __name__ == '__main__':
    main()
