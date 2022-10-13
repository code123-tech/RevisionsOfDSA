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

# Question 1


class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        minimum_time = 0
        curr_maxi_time = 0

        index = 0
        while index < len(colors):

            j = index
            sum_ = neededTime[j]
            curr_maxi_time = neededTime[j]

            while j < len(colors) - 1 and colors[j] == colors[j+1]:
                curr_maxi_time = max(curr_maxi_time, neededTime[j+1])
                sum_ += neededTime[j+1]
                j += 1

            if j == index:
                sum_ = 0
                curr_maxi_time = 0

            index = j + 1
            minimum_time += (sum_ - curr_maxi_time)

        return minimum_time

        '''
        for i in range(len(colors)):
            if i > 0 and colors[i] != colors[i-1]:
                curr_maxi_time = 0
                
            minimum_time += min(curr_maxi_time, neededTime[i])
            curr_maxi_time = max(curr_maxi_time, neededTime[i])
        
        return minimum_time
        
        '''


'''
    N = len(colors) = len(neededTime)
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 2


class Solution:
    def minimizeXor(self, num1: int, num2: int) -> int:
        # Step 1: Count number of set bits in num2
        set_bits = bin(num2)[2:].count("1")

        # Step 2: calculate the binary of num1 till 32 bit, and create result of 32 bit
        num1_bits_array = list(bin(num1)[2:])
        num1_bits_array = ["0"]*(32-len(num1_bits_array)) + num1_bits_array
        result = ["0"]*32

        # Step 3: Iterate over the bits of num1 and set the bits of result
        for i in range(len(num1_bits_array)):
            if num1_bits_array[i] == "1" and set_bits > 0:
                result[i] = "1"
                set_bits -= 1

        # Step 4: number of set bits are still remaining, then set those bit which are "0" in result from reverse traversel.
        index = 31
        while set_bits > 0 and index > -1:
            if result[index] == "0":
                result[index] = "1"
                set_bits -= 1
            index -= 1

        # Step 5: Convert the result to integer
        return int("".join(result), 2)


# Question 3
class Solution:
    def maxSum(self, grid: List[List[int]]) -> int:
        maximum_sum = 0
        for row in range(len(grid)-2):
            for col in range(len(grid[0])-2):
                current_sum = grid[row][col] + grid[row][col +
                                                         1] + grid[row][col+2]  # First row sum
                current_sum += grid[row+1][col+1]  # Middle number sum
                current_sum += grid[row+2][col] + grid[row +
                                                       2][col+1] + grid[row+2][col+2]  # Last
                maximum_sum = max(maximum_sum, current_sum)
        return maximum_sum


'''
    N = len(grid)
    M = len(grid[0])
    Time Complexity: O(N*M)
    Space Complexity: O(1)
'''

# Question 4


class Solution:
    def xorAllNums(self, nums1: List[int], nums2: List[int]) -> int:
        xor = 0
        len1 = len(nums1)
        len2 = len(nums2)
        isLen1Odd = len1 & 1
        isLen2Odd = len2 & 1

        for i in range(len(nums1)):
            if isLen2Odd:
                xor ^= nums1[i]
        for i in range(len(nums2)):
            if isLen1Odd:
                xor ^= nums2[i]
        return xor


'''
    N = len(nums1)
    M = len(nums2)
    Time Complexity: O(N+M)
    Space Complexity: O(1)
'''

# Question 5


class LUPrefix:

    def __init__(self, n: int):
        self.highest = 0
        self.video = set()

    def upload(self, video: int) -> None:
        self.video.add(video)
        while self.highest + 1 in self.video:
            self.highest += 1

    def longest(self) -> int:
        return self.highest


'''
    N = len(video)
    Time Complexity: O(N)
    Space Complexity: O(N)
'''


class Solution:
    def addOneRow(self, root: Optional[TreeNode], val: int, d: int) -> Optional[TreeNode]:
        if d == 1:
            if not root:
                return TreeNode(val)
            else:
                newroot = TreeNode(val)
                newroot.left = root
                return newroot

        queue = deque()
        queue.append([root, 1])

        d = d-1
        while queue:
            node, depth = queue.popleft()

            if depth == d:
                l, r = node.left, node.right
                node.left = TreeNode(val)
                node.right = TreeNode(val)
                node.left.left = l
                node.right.right = r

            if node.left:
                queue.append([node.left, depth+1])
            if node.right:
                queue.append([node.right, depth+1])

        return root


'''
    N = number of nodes in the tree
    Time Complexity: O(N)
    Space Complexity: O(N)
    
'''

# Question 6

'''
    Intution: the main intution is to inverse the node values at each of the odd level.
    So, We will use Level Order traversel for reaching level by level of all the nodes.

    Algorithm:
    1. We will use a queue for storing the nodes of the tree.
    2. We will use a variable "level" for storing the current level of the tree.
    3. As we catch level as odd, we will iterate over all the nodes present in queue, and reverse all the nodes values.
    4. We will return the root node at the end.
'''


class Solution:
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        queue = deque()
        queue.append(root)
        level = 0

        while queue:
            if level & 1:
                left = 0
                right = len(queue)-1
                while left <= right:
                    queue[left].val, queue[right].val = queue[right].val, queue[left].val
                    left += 1
                    right -= 1

            for _ in range(len(queue)):
                node = queue.popleft()

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            level = 1-level

        return root


'''
    N = number of nodes in the tree
    Time Complexity: O(N)
    Space Complexity: O(N)
'''

# Question 7


class Solution:
    def longestContinuousSubstring(self, s: str) -> int:
        cur = 1
        res = 1
        for i in range(1, len(s)):
            if ord(s[i])-ord(s[i-1]) == 1:
                cur = cur+1
                res = max(cur, res)
            else:
                cur = 1
        return res


'''
    N = len(s)
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 8


class Solution:
    def matchPlayersAndTrainers(self, players: List[int], trainers: List[int]) -> int:

        max_players_taken = 0
        players.sort()
        trainers.sort()
        i, j = len(players)-1, len(trainers)-1

        while i > -1 and j > -1:
            if players[i] > trainers[j]:
                i -= 1
            else:
                max_players_taken += 1
                i -= 1
                j -= 1

        return max_players_taken


'''
    N = len(players)
    M = len(trainers)
    Time Complexity: O(NlogN+MlogM)
    Space Complexity: O(1)
'''

# Question 9


class Solution:
    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        maximum = max(nums)
        maxi_bits = bin(maximum)[2:]
        number_of_bits = len(maxi_bits)

        bit_one_available = [float('-inf')]*number_of_bits

        for i in range(len(nums)-1, -1, -1):
            current = nums[i]
            current_bits = bin(current)[2:]
            current_bits = "0" * \
                (number_of_bits-len(current_bits)) + current_bits

            index = float('-inf')

            j = number_of_bits - 1

            while j > -1:
                if current_bits[j] == "0":
                    index = max(index, bit_one_available[number_of_bits-j-1])
                    if number_of_bits == "0":
                        index = max(index, j)
                else:
                    bit_one_available[number_of_bits-j-1] = i

                j -= 1

            if index == float('-inf'):
                nums[i] = 1
            else:
                nums[i] = index - i + 1

        return nums


'''
    N = len(nums)
    Time Complexity: O(N*32)
    Space Complexity: O(32) = O(1)
'''

# Question 10: Time Based Key-Value Store


class TimeMap:
    def __init__(self):
        self.dicti = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:

        self.dicti[key].append([value, timestamp])

    def get(self, key: str, timestamp: int) -> str:

        lst = self.dicti[key]

        left = 0
        right = len(lst)
        while left < right:
            mid = (left+right)//2
            if lst[mid][1] <= timestamp:
                left = mid+1
            else:
                right = mid
        return "" if right == 0 else lst[right-1][0]


'''
    N = number of keys
    Maximum Calls = 2*10^5
    Maximum 2*10^5 - 1 calls to set method, and suppose same key is called again and again, then we will have 2*10^5 - 1 calls to set method.
    so, dicti[key] = lst.  # len(lst) = 10^5

    So, R = 10^5
    Time Complexity: O(logR)
'''

# Question 11:


class Solution:
    def minGroups(self, intervals: List[List[int]]) -> int:

        groups = 0
        current = 0
        temp = []
        for a, b in intervals:
            temp.append([a, 1])  # entering into the meeting
            temp.append([b+1, -1])  # Leaving the meeting at b+1

        temp.sort()
        for el, diff in temp:
            current += diff
            groups = max(groups, current)

        return groups


'''
    N = len(intervals)
    total_elements in A = N*2
    len(A) = A
    Time Complexity: O(AlogA)
    Space Complexity: O(N*2)
'''

# Question 12: My Caleder III


class MyCalendarThree:
    def __init__(self):
        self.dicti = SortedDict()

    def book(self, start: int, end: int) -> int:
        if start not in self.dicti:
            self.dicti[start] = 1
        else:
            self.dicti[start] += 1

        if end not in self.dicti:
            self.dicti[end] = -1
        else:
            self.dicti[end] -= 1

        res = cur = 0
        for key, val in self.dicti.items():
            cur += val
            res = max(res, cur)
        return res


# Question 13: 3 Sum Closest
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        res = nums[0] + nums[1] + nums[2]

        for i in range(len(nums)-2):
            left = i+1
            right = len(nums)-1
            while left < right:
                total = nums[i]+nums[left]+nums[right]
                if total == target:
                    return total

                if abs(target-total) < abs(target-res):
                    res = total
                if total < target:
                    left += 1
                else:
                    right -= 1
        return res


'''
    N = len(nums)
    Time Complexity: O(N^2)
    Space Complexity: O(1)
'''

# Question 14: Break a Palindrome


class Solution:
    def breakPalindrome(self, p: str) -> str:
        n = len(p)
        if n == 1:
            return ""

        left, right = 0, n-1
        while left < right:
            if p[left] == 'a':
                left, right = left+1, right-1
                continue
            break

        if left >= right:
            return p[:n-1] + "b"

        return p[:left] + "a" + p[left+1:]


'''
    N = len(p)
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 15: Minimum number of jumps


class Solution:
    def minJumps(self, arr, n):
        if n == 1:
            return 0
        if arr[0] == 0:
            return -1

        max_reach = arr[0]
        steps = arr[0]
        jumps = 1

        for i in range(1, n):
            if i == n-1:
                return jumps

            max_reach = max(max_reach, i+arr[i])
            steps -= 1

            if steps == 0:
                jumps += 1
                if i >= max_reach:
                    return -1
                steps = max_reach-i

        return -1

# Question 16: Delete Node in a Linked List


class Solution:
    def deleteNode(self, node):
        prev = None
        while node.next is not None:
            node.val = node.next.val
            prev = node
            node = node.next
        prev.next = None


'''
    N = number of nodes in the linked list
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 17: Rotate Array


class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        n = len(nums)
        k = k % n
        nums[:n-k] = nums[:n-k][::-1]
        nums[n-k:] = nums[n-k:][::-1]
        left = 0
        right = len(nums) - 1
        while left <= right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1


'''
    N = len(nums)
    Time Complexity: O(N)
    Space Complexity: O(1)   
'''
