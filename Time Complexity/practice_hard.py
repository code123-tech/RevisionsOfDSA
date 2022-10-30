from bisect import bisect_right
import copy
from functools import lru_cache
import heapq
from itertools import accumulate
from optparse import Option
from tokenize import String
from typing import List, Optional
from collections import Counter, defaultdict, deque
from sortedcontainers import SortedDict
import re
from data_structures.Trie import Trie
from data_structures.UnionFind import UnionFind


class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = self.right = None


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Question 1: Shortest Path in a Grid with Obstacles Elimination
# https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])

        steps = 0  # steps taken
        queue = deque()
        queue.append([0, 0, k])  # current_x, current_y, remaining_elimination

        remaining_steps = [[float('-inf') for j in range(n)] for i in range(m)]
        remaining_steps[0][0] = k

        while queue:
            for _ in range(len(queue)):
                current_x, current_y, current_elimination = queue.popleft()

                if current_x == m-1 and current_y == n-1:
                    return steps
                for dir_x, dir_y in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                    next_x, next_y = current_x + dir_x, current_y + dir_y
                    if next_x < 0 or next_x >= m or next_y < 0 or next_y >= n:
                        continue

                    remaining_elimination = current_elimination - \
                        grid[next_x][next_y]

                    if remaining_elimination >= 0 and remaining_steps[next_x][next_y] < remaining_elimination:
                        queue.append([next_x, next_y, remaining_elimination])

                        remaining_steps[next_x][next_y] = remaining_elimination

            steps += 1

        return -1


'''
    Time Complexity: O(m*n*k)
    Space Complexity: O(m*n)
'''
