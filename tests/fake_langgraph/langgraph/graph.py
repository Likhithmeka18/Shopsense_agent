import typing
from . import types as _t

START, END = "__start__", "__end__"


def _reducers(schema):
    out = {}
    try:
        hints = typing.get_type_hints(schema, include_extras=True)
    except Exception:
        hints = {}
    for k, h in hints.items():
        if typing.get_origin(h) is typing.Annotated:
            for meta in h.__metadata__:
                if callable(meta):
                    out[k] = meta
    return out


class StateGraph:
    def __init__(self, schema):
        self.schema, self.nodes, self.edges, self.cond = schema, {}, {}, {}

    def add_node(self, name, fn):
        assert name not in self.nodes and name not in (START, END), f"duplicate/reserved node {name}"
        self.nodes[name] = fn

    def add_edge(self, a, b):
        assert a == START or a in self.nodes, f"edge from unknown node {a}"
        assert b == END or b in self.nodes, f"edge to unknown node {b}"
        self.edges.setdefault(a, []).append(b)

    def add_conditional_edges(self, a, fn, mapping=None):
        assert a in self.nodes, f"conditional edge from unknown node {a}"
        if mapping:
            for dest in mapping.values():
                assert dest == END or dest in self.nodes, f"conditional edge to unknown node {dest}"
        self.cond[a] = (fn, mapping)

    def compile(self, checkpointer=None):
        assert START in self.edges, "no START edge"
        return Compiled(self, checkpointer)


class Compiled:
    def __init__(self, g, saver):
        self.g, self.saver, self.red = g, saver, _reducers(g.schema)

    def _merge(self, state, update):
        out = dict(state)
        allowed = set(typing.get_type_hints(self.g.schema)) if update is not None else set()
        for k, v in (update or {}).items():
            assert k in allowed, f"node returned {k!r}, which is not a key of {self.g.schema.__name__} (real LangGraph raises InvalidUpdateError)"
            out[k] = self.red[k](state.get(k, []), v) if k in self.red else v
        return out

    def _next(self, node, state):
        if node in self.g.cond:
            fn, mapping = self.g.cond[node]
            key = fn(state)
            return mapping[key] if mapping else key
        dests = self.g.edges.get(node, [])
        assert len(dests) == 1, f"node {node} must have exactly one successor in this fake (has {dests})"
        return dests[0]

    def get_state(self, config):
        import types as _types
        ck = self.saver.get(config["configurable"]["thread_id"]) if self.saver else None
        return _types.SimpleNamespace(next=(ck["pending"],) if ck and ck.get("pending") else ())

    def invoke(self, inp, config):
        tid = config["configurable"]["thread_id"]
        ck = self.saver.get(tid) if self.saver else None
        if isinstance(inp, _t.Command):
            assert ck and ck.get("pending"), "nothing to resume on this thread"
            state, node, resume = ck["state"], ck["pending"], inp.resume
        else:
            state = self._merge(ck["state"] if ck else {}, inp)
            node, resume = self.g.edges[START][0], _t._NO_RESUME
        for _ in range(200):
            if node == END:
                break
            _t._set_resume(resume)
            try:
                update = self.g.nodes[node](state)
            except _t.GraphInterrupt as gi:
                if self.saver:
                    self.saver.put(tid, {"state": state, "pending": node})
                return {**state, "__interrupt__": [_t.Interrupt(gi.value)]}
            finally:
                _t._set_resume(_t._NO_RESUME)
            resume = _t._NO_RESUME
            state = self._merge(state, update)
            node = self._next(node, state)
        else:
            raise RuntimeError("graph did not terminate (the real library would raise GraphRecursionError)")
        if self.saver:
            self.saver.put(tid, {"state": state, "pending": None})
        return state
