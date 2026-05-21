from collections import defaultdict


class SemanticRegionGraph:
    def __init__(self) -> None:
        self._adjacency: dict[str, set[str]] = defaultdict(set)

    def connect(self, a, b):
        self._adjacency[a].add(b)
        self._adjacency[b].add(a)

    def neighbors(self, node: str) -> set[str]:
        return set(self._adjacency.get(node, set()))
