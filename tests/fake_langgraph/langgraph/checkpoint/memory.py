class InMemorySaver:
    def __init__(self):
        self.data = {}

    def get(self, tid):
        return self.data.get(tid)

    def put(self, tid, ck):
        self.data[tid] = ck
