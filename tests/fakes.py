class FakeDispatcher:
    def __init__(self, tier): self.tier = tier; self.calls = []
    def dispatch(self, step, run, payload, callback_url):
        self.calls.append((step.name, run.id, payload, callback_url))

class FakeReporter:
    def __init__(self):
        self.progress_calls = []; self.dub_results = []
        self.transcript_results = []; self.failures = []
    def progress(self, run, label, pct=None): self.progress_calls.append((run.id, label, pct))
    def dub_result(self, run): self.dub_results.append(run.id)
    def transcript_result(self, run): self.transcript_results.append(run.id)
    def failed(self, run, step, error): self.failures.append((run.id, step, error))
