"""A ~120-line stand-in for the parts of LangGraph that ShopSense uses, for OFFLINE wiring checks only.
It follows the documented semantics the project relies on (nodes re-run from their top on resume, `interrupt()` returns the
resume value on the second run, `Annotated` reducers, conditional edges, checkpoints per thread_id) but it is NOT LangGraph:
a pass here proves the node names / edge maps / state keys line up; the real library is still checked by the live demos."""
