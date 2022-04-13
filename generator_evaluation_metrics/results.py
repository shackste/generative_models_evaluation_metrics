import json
from pprint import pformat

class Results:
    results = {}

    def _key(self, model, encoder, metric):
        return f"{encoder} {metric} {model}"

    def append(self, value, *, model: str = "", encoder: str = "", metric: str = ""):
        assert model != "", "provide model name"
        assert encoder != "", "provide model name"
        assert metric != "", "provide model name"
        self.results[self._key(model, encoder, metric)] = value

    def get_value(self,  *, model: str = "", encoder: str = "", metric: str = ""):
        assert model != "", "provide model name"
        assert encoder != "", "provide model name"
        assert metric != "", "provide model name"
        return self.results[self._key(model, encoder, metric)]

    ## add nice representation functions
    def __repr__(self):
        return pformat(self.results, indent=4)

