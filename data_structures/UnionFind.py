class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0]*(n)

    def find(self, u):
        if self.parent[u] == u:
            return u

        return self.find(self.parent[u])

    def union(self, u, v):
        x, y = self.find(u), self.find(v)
        if self.rank[x] < self.rank[y]:
            self.parent[x] = y
        elif self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[y] = x
            self.rank[x] += 1
