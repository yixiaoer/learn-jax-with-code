from __future__ import annotations

from typing import Any, Callable, NamedTuple

import numpy as np

class Context(NamedTuple):
    id_gen: Callable
    variables: dict

class Tracer:
    def __init__(self, arr: np.ndarray, ctx: Context, id_: str | None = None, records: list[tuple] | None = None) -> None:
        self.arr = arr
        self.ctx = ctx
        if id_ is None:
            id_ = ctx.id_gen()
        ctx.variables[id_] = arr
        self.id = id_
        self.records = records

    def __add__(self, other: Any) -> Tracer:
        arr, ctx, records, id_ = self.arr, self.ctx, self.records, self.id
        is_float = isinstance(other, float)
        is_Tracer = isinstance(other, Tracer)
        if is_float or is_Tracer:
            id_new = ctx.id_gen()
            other_ = other if is_float else other.id
            add_res = arr + other if is_float else arr + other.arr
            if records is not None:
                records.append(('__add__', id_, (other_,), {}, id_new))
            return Tracer(add_res, ctx, id_=id_new, records=records)
        raise NotImplementedError('Not supported.')
    
    def __matmul__(self, other: Any) -> Tracer:
        arr, ctx, records, id_ = self.arr, self.ctx, self.records, self.id
        is_ndarray = isinstance(other, np.ndarray)
        is_Tracer = isinstance(other, Tracer)
        if is_ndarray or is_Tracer:
            id_new = ctx.id_gen()
            other_ = other if is_ndarray else other.id
            mat_res = arr @ other if is_ndarray else arr @ other.arr
            if records is not None:
                records.append(('__matmul__', id_, (other_,), {}, id_new))
            return Tracer(mat_res, ctx, id_=id_new, records=records)
        raise NotImplementedError('Not supported.')

    def reshape(self, *args, **kwargs) -> Tracer:
        arr, ctx, records, id_ = self.arr, self.ctx, self.records, self.id
        id_new = ctx.id_gen()
        if records is not None:
            records.append(('reshape', id_, args, kwargs, id_new))
        return Tracer(arr.reshape(*args, **kwargs), ctx, id_=id_new, records=records)

    def sum(self, *args, **kwargs) -> Tracer:
        arr, ctx, records, id_ = self.arr, self.ctx, self.records, self.id
        id_new = ctx.id_gen()
        if records is not None:
            records.append(('sum', id_, args, kwargs, id_new))
        return Tracer(np.array(arr.sum()), ctx, id_=id_new, records=records)

    def __repr__(self) -> str:
        arr, id_ = self.arr, self.id
        return f'Tracer object {repr(id_)}:\n{repr(arr)}'

class Variable:
    def __init__(self, id_: int) -> None:
        self.id = id_

    def __repr__(self) -> str:
        return f'v{self.id}'

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.id == other.id

class IDGenerator:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> Variable:
        out = Variable(self.count)
        self.count += 1
        return out

def make_tracer_if_is_np_arr(x, ctx, /, records: list[tuple] | None = None, i: int | None = None, static_argnums=(), k=None, static_argnames=()) -> Any:
    if not isinstance(x, np.ndarray):
        return x
    if i is not None and i in static_argnums:
        return x
    if k is not None and k in static_argnames:
        return x
    return Tracer(x, ctx, records=records)

def recover_arg(arg: Any, variables: dict[Variable, Any]) -> Any:
    if not isinstance(arg, Variable):
        return arg
    return variables[arg]

def recover_args(args: list, variables: dict[Variable, Any]) -> list:
    return [recover_arg(arg, variables) for arg in args]

def recover_kwargs(kwargs: dict[str, Any], variables: dict[Variable, Any]) -> dict[str, Any]:
    return {k: recover_arg(v, variables) for k, v in kwargs.items()}

class trace:
    def __init__(self, f: Callable, static_argnums=(), static_argnames=()) -> None:
        self.f = f
        self.static_argnums = static_argnums
        self.static_argnames = static_argnames
        self.records = None

    def __call__(self, *args, **kwargs) -> np.ndarray:
        f, static_argnums, static_argnames = self.f, self.static_argnums, self.static_argnames
        id_gen = IDGenerator()
        variables = {}
        ctx = Context(id_gen, variables)
        if self.records is None:
            records = []
            args = [make_tracer_if_is_np_arr(x, ctx, records=records, i=i, static_argnums=static_argnums) for i, x in enumerate(args)]
            kwargs = {k: make_tracer_if_is_np_arr(v, ctx, records=records, k=k, static_argnames=static_argnames) for k, v in kwargs.items()}
            out = f(*args, **kwargs)
            self.records = records
            return out.arr
        else:
            records = self.records
            [make_tracer_if_is_np_arr(x, ctx, records=None, i=i, static_argnums=static_argnums) for i, x in enumerate(args)]
            {k: make_tracer_if_is_np_arr(v, ctx, records=None, k=k, static_argnames=static_argnames) for k, v in kwargs.items()}
            for op, obj, args, kwargs, id_new in records:
                obj = recover_arg(obj, variables)
                args = recover_args(args, variables)
                kwargs = recover_kwargs(kwargs, variables)
                match op:
                    case 'reshape'   : out = obj.reshape(*args, **kwargs)
                    case '__add__'   : out = obj.__add__(*args, **kwargs)
                    case '__matmul__': out = obj.__matmul__(*args, **kwargs)
                    case 'sum'       : out = obj.sum(*args, **kwargs)
                    case _           : raise NotImplementedError(f'Not supported op: {op}')
                id_new = id_gen()
                variables[id_new] = out
            return out

def main():
    # test calculation function
    def f(a, b, /, *, c):
        return a.reshape((24, 2), order='C') @ b.reshape((2, 9)) + c.sum() + 4.

    a = np.random.randn(12, 4)
    b = np.random.randn(3, 6)
    c = np.random.randn(2)

    out1 = f(a, b, c=c)

    f_traced = trace(f, static_argnums=(1,), static_argnames=('c',))
    
    out2 = f_traced(a, b, c=c)
    for record in f_traced.records:
        print(record)
    out3 = f_traced(a, b, c=c)

    print(np.all(out1 == out2) and np.all(out1 == out3))

if __name__ == '__main__':
    main()
