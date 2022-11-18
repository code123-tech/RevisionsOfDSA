from bisect import bisect_right, bisect_left
# import copy
from functools import lru_cache
import heapq
# from itertools import accumulate
# from optparse import Option
from tokenize import String
from typing import List, Optional
from collections import Counter, defaultdict, deque
from sortedcontainers import SortedDict
# import re
from data_structures.Trie import Trie
from data_structures.UnionFind import UnionFind
import math


class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = self.right = None


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
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

# Question 18: Delete the Middle Node of a Linked List


class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head.next is None:
            return None

        slow = head
        fast = head
        while fast.next and fast.next.next and fast.next.next.next:
            slow = slow.next
            fast = fast.next.next

        slow.next = slow.next.next
        return head


'''
    N = number of nodes in the linked list
    Time Complexity: O(N)
    Space Complexity: O(1)
'''


# Question 19: Integer to Roman
class Solution:
    def intToRoman(self, num: int) -> str:
        digits = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        romanNum = ["M", "CM", "D", "CD", "C", "XC",
                    "L", "XL", "X", "IX", "V", "IV", "I"]
        string = ""
        for i in range(len(digits)):
            while(num >= digits[i]):
                string += romanNum[i]
                num -= digits[i]
            if(num <= 0):
                break
        return string


'''
    Time Complexity: O(1)
    Space Complexity: O(1)
'''

# Question 20: Find the element that appears once
'''
Given a sorted array A[] of N positive integers having all the numbers occurring exactly twice, except for one number which will occur only once. Find the number occurring only once.
'''


class Solution:
    def search(self, A, N):
        low = 0
        high = N-1
        while low <= high:
            mid = (low+high)//2
            if mid % 2 == 0:
                if mid+1 < N and A[mid] == A[mid+1]:
                    low = mid+2
                elif mid-1 >= 0 and A[mid] == A[mid-1]:
                    high = mid-2
                else:
                    return A[mid]
            else:
                if mid+1 < N and A[mid] == A[mid+1]:
                    high = mid-1
                elif mid-1 >= 0 and A[mid] == A[mid-1]:
                    low = mid+1
                else:
                    return A[mid]
        return -1


'''
    Time Complexity: O(logN)
    Space Complexity: O(1)
'''

# Question 21: Continuous Subarray Sum


class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        dicti = {0: -1}
        sum_ = 0
        for i in range(len(nums)):
            sum_ = (sum_ + nums[i]) % k
            if sum_ in dicti:
                if i - dicti[sum_] > 1:
                    return True

            else:
                dicti[sum_] = i

        return False


'''
    N = len(nums)
    Time Complexity: O(N)
    Space Complexity: O(k)
'''


# Question 22: Image Overlap
# Link: https://leetcode.com/problems/image-overlap/
class Solution:
    def largestOverlap(self, A: List[List[int]], B: List[List[int]]) -> int:

        dim = len(A)

        def shift_and_count(x_shift, y_shift, M, R):
            left_shift_count, right_shift_count = 0, 0
            for r_row, m_row in enumerate(range(y_shift, dim)):
                for r_col, m_col in enumerate(range(x_shift, dim)):
                    if M[m_row][m_col] == 1 and M[m_row][m_col] == R[r_row][r_col]:
                        left_shift_count += 1
                    if M[m_row][r_col] == 1 and M[m_row][r_col] == R[r_row][m_col]:
                        right_shift_count += 1

            return max(left_shift_count, right_shift_count)

        max_overlaps = 0
        for y_shift in range(0, dim):
            for x_shift in range(0, dim):
                max_overlaps = max(
                    max_overlaps, shift_and_count(x_shift, y_shift, A, B))
                max_overlaps = max(
                    max_overlaps, shift_and_count(x_shift, y_shift, B, A))

        return max_overlaps


'''
    Time Complexity: O(N^4)
    Space Complexity: O(1)
'''


# Question 23: Sort Colors
# Link: https://leetcode.com/problems/sort-colors/
class Solution:
    def sortColors(self, nums: List[int]) -> None:

        low = 0
        mid = 0
        high = len(nums) - 1

        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1
            else:
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1


'''
    Time Complexity: O(N)
    Space Complexity: O(1)

'''

# Question 24: Majority Element
# Link: https://leetcode.com/problems/majority-element/
# Similar to: Boyer-Moore Voting Algorithm


class Solution:
    def majorityElement(self, A, N):
        # Your code here
        count = 0
        candidate = None
        for num in A:
            if count == 0:
                candidate = num
            count = (count+1 if candidate == num else count - 1)

        cand_count = A.count(candidate)
        if cand_count > N//2:
            return candidate
        return -1


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 25: Group Anagrams
# Link: https://leetcode.com/problems/group-anagrams/


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

        def countCharacters(s):
            count = [0]*26
            for ch in s:
                index = ord(ch) - 97
                count[index] += 1

            return '-'.join(map(str, count))

        dicti = defaultdict(list)
        for i in strs:
            dicti[countCharacters(i)].append(i)

        ans = []
        for key, value in dicti.items():
            ans.append(dicti[key])
        return ans


'''
    Time Complexity: O(NK)
    Space Complexity: O(NK)
'''


# Question 26: Stock buy and sell
# Link: https://practice.geeksforgeeks.org/problems/stock-buy-and-sell2615/0
def stockBuySell(price, n):
    buyDay = 0
    sellDay = 0
    isProfitGained = False

    for i in range(1, n):
        if price[i] >= price[i-1] and price[i] != price[buyDay]:
            sellDay += 1
            isProfitGained = True
        else:
            if buyDay != sellDay:
                print("(" + str(buyDay) + " " + str(sellDay) + ")", end=" ")
            buyDay = i
            sellDay = i

    if buyDay != n - 1 and sellDay == n-1:
        print("(" + str(buyDay) + " " + str(sellDay) + ")", end=" ")

    if isProfitGained == False:
        print("No Profit")
    else:
        print()


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 27: Longest Consecutive Sequence
# Link: https://leetcode.com/problems/longest-consecutive-sequence/


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        keys = set(nums)
        best = 0
        for key in keys:
            count = 0
            if key-1 not in keys:
                count += 1
                y = key+1
                while y in keys:
                    y += 1
                    count += 1
                best = max(best, count)
        return best


'''
    Time Complexity: O(N)
    Space Complexity: O(N)
'''

# Question 28: Rotate image by 90 degree (Anti Clock wise)
# Link: https://practice.geeksforgeeks.org/problems/rotate-by-90-degree-1587115621/1


class Solution:
    def rotateby90(self, a, n):
        # code here
        a.reverse()
        for i in range(n):
            for j in range(n-i-1):
                a[i][j], a[n-j-1][n-i-1] = a[n-j-1][n-i-1], a[i][j]


'''
    Time Complexity: O(N^2)
    Space Complexity: O(1)
'''

# Question 29: Earliest Possible Day of Full Bloom
# Link: https://leetcode.com/problems/earliest-possible-day-of-full-bloom/


class Solution:
    def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
        # if len(plantTime) == 1:
        #     return plantTime[0] + growTime[0]

        seed_grow = [[plantTime[i], growTime[i]]
                     for i in range(len(plantTime))]

        seed_grow.sort(key=lambda x: (-x[1], x[0]))

        result = 0
        current = 0
        for seed, grow in seed_grow:
            current += seed
            result = max(result, current + grow)

        return result


'''
    Time Complexity: O(NlogN)
    Space Complexity: O(N)
'''

# Question 30: Majority Element II
# Link: https://leetcode.com/problems/majority-element-ii/
# intution: https://leetcode.com/problems/majority-element-ii/discuss/543672/BoyerMoore-majority-vote-algorithm-EXPLAINED


class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        num1 = num2 = None
        count1 = count2 = 0
        for i in range(len(nums)):
            if nums[i] == num1:
                count1 += 1
            elif nums[i] == num2:
                count2 += 1
            elif count1 == 0:
                count1 = 1
                num1 = nums[i]
            elif count2 == 0:
                count2 = 1
                num2 = nums[i]
            else:
                count1 -= 1
                count2 -= 1
        real_count_num1 = nums.count(num1)
        real_count_num2 = nums.count(num2)
        result = []
        if real_count_num1 > len(nums)//3:
            result.append(num1)
        if real_count_num2 > len(nums)//3:
            result.append(num2)
        return result


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''
# Question 31: 4Sum
# Link: https://leetcode.com/problems/4sum/


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        result = set()
        nums.sort()

        for i in range(len(nums)-2):
            for j in range(i+1, len(nums)-2):
                left = j + 1
                right = len(nums) - 1
                while left < right:
                    curr_sum = nums[i] + nums[j] + nums[left] + nums[right]

                    if curr_sum > target:
                        right -= 1
                    elif curr_sum < target:
                        left += 1
                    else:
                        result.add((nums[i], nums[j], nums[left], nums[right]))
                        left += 1
                        right -= 1
        return result


'''
    Time Complexity: O(N^3)
    Space Complexity: O(N)
'''

# Question 32: Merge Intervals
# Link: https://leetcode.com/problems/merge-intervals/


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:

        intervals.sort(key=lambda x: x[0])
        result = [intervals[0]]
        for lst in intervals[1:]:
            if result[-1][-1] < lst[0]:
                result.append(lst)
            else:
                result[-1][-1] = max(result[-1][-1], lst[-1])
        return result


'''
    Time Complexity: O(NlogN)
    Space Complexity: O(N)
'''

# Question 33: Subsets with XOR value
# Link: https://practice.geeksforgeeks.org/problems/subsets-with-xor-value2023/1


class Solution:
    def subsetXOR(self, arr, N, K):
        # code here

        @lru_cache(None)
        def recur(index, xor):
            if index == N:
                return xor == K

            left = recur(index+1, xor ^ arr[index])
            right = recur(index+1, xor)

            return left + right
        return recur(0, 0)


'''
    Time Complexity: O(N*M)
    Space Complexity: O(N*M)
'''
# Question 34: Where Will the Ball Fall
# Link: https://leetcode.com/problems/where-will-the-ball-fall/


class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        rows = len(grid)
        cols = len(grid[0])

        ans = [-1]*cols

        def dfs(ball, row, col):
            if col >= cols or col < 0:
                return

            if row == rows:
                ans[ball] = col
                return

            if (col == 0 and grid[row][col] == -1) or (col == cols-1 and grid[row][col] == 1):
                return

            if col+1 < cols and grid[row][col] == 1 and grid[row][col+1] != -1:
                dfs(ball, row+1, col+1)

            elif col-1 >= 0 and grid[row][col] == -1 and grid[row][col-1] != 1:
                dfs(ball, row+1, col-1)

        for i in range(cols):
            dfs(i, 0, i)

        return ans


'''
    Time Complexity: O(N^2)
    Space Complexity: O(N)
'''

# Question 35: Merge Without Extra Space
# Link: https://practice.geeksforgeeks.org/problems/merge-two-sorted-arrays-1587115620/1

'''
    Gap method is being used here:
    1. firstly gap value is  (n+m)//2
        * Compare arr1[i] and arr1[i+gap] until i+gap < n
        * Compare arr1[i] and arr2[j] with j - i = gap  until j < m and i < n
        * Compare arr2[j] and arr2[j+gap]   until j+gap < m
    3. then change the gap value to gap//2
    4. repeat the process until gap is greater than 0
'''


class Solution:

    # Function to merge the arrays.
    def merge(self, arr1, arr2, n, m):
        def findGap(x):
            if x <= 1:
                return 0
            return (x//2) + (x % 2)

        total = n + m
        gap = findGap(total)

        while gap > 0:

            # gap within first array
            i = 0
            while i + gap < n:
                if arr1[i] > arr1[i+gap]:
                    arr1[i], arr1[i+gap] = arr1[i+gap], arr1[i]
                i += 1

            # gap into both array
            j = gap - n if gap > n else 0
            while i < n and j < m:
                if arr1[i] > arr2[j]:
                    arr1[i], arr2[j] = arr2[j], arr1[i]

                i += 1
                j += 1

            # gap into last array
            if j < m:
                j = 0
                while j + gap < m:
                    if arr2[j] > arr2[j+gap]:
                        arr2[j], arr2[j+gap] = arr2[j+gap], arr2[j]
                    j += 1

            gap = findGap(gap)


'''
    K = n + m 
    Time Complexity: O(KlogK)
    Space Complexity: O(1)
'''


# Question 36: Find the repeating and missing number
# Link: https://practice.geeksforgeeks.org/problems/find-missing-and-repeating2512/1
'''
    first approach: Iterate from 1 to n, and check if i is repeating or missing in arr
            Time Complexity: O(N^2), Space Complexity: O(1)
    second approach: Sorting the array, and the find missing and repeating
            Time Complexity: O(NlogN), Space Complexity: O(1)
    third Approach: Using counting of each element in the array
            Time Complexity: O(N), Space Complexity: O(N)
    fourth Approach: Using sum upto N number and sum of square upto n number
            Time Complexity: O(N), Space Complexity: O(1)
            Basically in this approach we are using the formula of sum of n natural number
            and sum of square of n natural number
            sum = n(n+1)/2
            sum of square = n(n+1)(2n+1)/6 

            1. remaining_sum =  sum - sum_of_array (remaining_sum = missing_number - repeating_number)
            2. remaining_sum_square = sum_of_square - sum_of_square_array (remaining_sum_square = missing_number^2 - repeating_number^2)

            dividing equation 2 by equation 1

               (missing_number^2 - repeating_number^2) =  remaining_sum_square
           =>  ---------------------------------------    --------------------
               (missing_number - repeating_number) =  remaining_sum

    fifth Approach: Using XOR
            Time Complexity: O(N), Space Complexity: O(1)
            
'''


class Solution:
    def findTwoElement(self, arr, n):
        # four approache
        sum_n = (n*(n+1))//2
        sum_sq = (n*(n+1)*(2*n+1))//6
        for i in range(n):
            sum_n -= arr[i]
            sum_sq -= (arr[i]*arr[i])
        miss = (sum_n + sum_sq//sum_n)//2
        return [miss-sum_n, miss]

        # fifth approach
        xor = 0
        # 1. find xor of all array elements
        for i in range(n):
            xor ^= arr[i]

        # 2. find xor of all elements from 1 to n, and take its xor with above xor, so you will get
        # X^Y = xor  (x is missing number, y is repeating number)
        for i in range(1, n+1):
            xor ^= i

        # 3. check rightmost set bit in the X^y = xor
        # which will tell us that either of the number has set bit of rightmost index.
        right_most_set_bit = xor & ~(xor-1)
        x = 0
        y = 0

        # 4. take all the number in x bucket which is having rightmost set bit.
        # . take all number in y bucket which is not having rightmost set bit.
        for i in range(n):
            if arr[i] & right_most_set_bit:
                x ^= arr[i]
            else:
                y ^= arr[i]

        # 5. now iterate from 1 to n again, and check which number is having rightmost set bit and which is not having.
        for i in range(1, n+1):
            if i & right_most_set_bit:
                x ^= i
            else:
                y ^= i

        # 6. now you find x and y as missing or repeating number.
        for i in range(n):
            if arr[i] == x:
                return [x, y]
        return [y, x]


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''


# Question 37: Maximum Product Subarray
# link: https://leetcode.com/problems/maximum-product-subarray/
class Solution:
    def maxProduct(self, arr: List[int]) -> int:
        n = len(arr)
        maxi = arr[0]
        mini = arr[0]
        result = arr[0]
        for i in range(1, n):
            c1 = maxi*arr[i]
            c2 = mini*arr[i]
            maxi = max(arr[i], max(c1, c2))
            mini = min(arr[i], min(c1, c2))
            result = max(result, maxi)

        return result


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 38: Find Peak Element
# Link: https://leetcode.com/problems/find-peak-element/


class Solution:
    def findPeakElement(self, arr: List[int]) -> int:

        def peakIndex(left, right):
            mid = left + (right-left)//2
            if ((mid == 0) or arr[mid-1] <= arr[mid]) and ((mid == len(arr)-1) or arr[mid+1] <= arr[mid]):
                return mid
            elif (mid > 0 and arr[mid-1] > arr[mid]):
                return peakIndex(left, mid-1)
            else:
                return peakIndex(mid+1, right)

        return peakIndex(0, len(arr)-1)


'''
    Time Complexity: O(logN)
    Space Complexity: O(1)
'''

# Question 39: Search in Rotated Sorted Array II
# Link: https://leetcode.com/problems/search-in-rotated-sorted-array-ii/


class Solution:
    def search(self, nums: List[int], key: int) -> bool:
        n = len(nums)
        l, h = 0, n-1
        # Code here
        while l <= h:
            # To avoid duplicates
            while l < h and nums[l] == nums[l + 1]:
                l += 1
            while l < h and nums[h] == nums[h - 1]:
                h -= 1

            mid = l + (h - l) // 2

            if nums[mid] == key:
                return 1

            if nums[mid] >= nums[l]:
                if nums[l] <= key < nums[mid]:
                    h = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < key <= nums[h]:
                    l = mid + 1
                else:
                    h = mid - 1

        return 0


'''
    Time Complexity: O(logN)
    Space Complexity: O(1)
'''
# Question 40: Median of two sorted arrays
# Link: https://practice.geeksforgeeks.org/problems/k-th-element-of-two-sorted-array1317/1


class Solution:
    def kthElement(self,  arr1, arr2, n, m, k):
        if n > m:
            return self.kthElement(arr2, arr1, m, n, k)

        left = max(0, k - m)
        right = min(k, n)
        while left <= right:
            mid1 = (left + right) >> 1   # mid1 of arr1
            mid2 = k - mid1  # mid2 of arr2

            l1 = float('-inf') if mid1 == 0 else arr1[mid1-1]
            r1 = float('inf') if mid1 >= n else arr1[mid2]

            l2 = float('-inf') if mid2 == 0 else arr2[mid2-1]
            r2 = float('inf') if mid2 >= m else arr2[mid2]

            if l1 <= r2 and l2 <= r1:
                return max(l1, l2)
            elif l1 > r2:
                right = mid1 - 1
            else:
                left = mid1 + 1


'''
    Time Complexity: O(log(min(N, M)))
    Space Complexity: O(1)
'''

# Question 41: Search in a 2D Matrix
# Link: https://leetcode.com/problems/search-a-2d-matrix/


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows, cols = len(matrix), len(matrix[0])
        row = 0
        col = len(matrix[0])-1

        while row < rows and col >= 0:

            if matrix[row][col] == target:
                return True
            if matrix[row][col] > target:
                col -= 1
            else:
                row += 1

        return False


'''
    Time Complexity: O(N+M)
    Space Complexity: O(1)
'''

# Question 42: Find a Peak Element II
# Link: https://leetcode.com/problems/find-a-peak-element-ii/


class Solution:
    def findPeakGrid(self, mat: List[List[int]]) -> List[int]:
        left = 0
        right = len(mat[0]) - 1

        while left <= right:
            maxRow = 0
            mid = left + (right-left)//2

            for row in range(1, len(mat)):
                if mat[maxRow][mid] <= mat[row][mid]:
                    maxRow = row

            isLeftBig = mid - 1 >= 0 and mat[maxRow][mid-1] >= mat[maxRow][mid]
            isRightBig = mid + \
                1 < len(mat[0]) and mat[maxRow][mid+1] >= mat[maxRow][mid]

            if not isLeftBig and not isRightBig:
                return [maxRow, mid]
            elif isLeftBig:
                right = mid - 1
            else:
                left = mid + 1


'''
    N = Number of rows
    M = Number of columns
    Time Complexity: O(NlogM)
    Space Complexity: O(1)
'''

# Question 43: Median in a row-wise sorted Matrix
# Link: https://practice.geeksforgeeks.org/problems/median-in-a-row-wise-sorted-matrix1527/1


class Solution:
    def median(self, matrix, R, C):
        '''
            As R and C are always odd, so  R*C is always odd
            and median is always the (R*C)/2 th element when we flatter the entire matrix.
            So, we can use binary search to find the median.

            We know, that median will lie in between minimum of entire matrix and maximum of entire matrix.

            So, each row is sorted. So we will check where can we fit the middle element in each row, and count the the number of elements greater than the middle element in each row. 
            if that count is less than equal to desired_count, then we have found the median.
        '''
        desired_count = (R*C)//2
        left = min([matrix[i][0] for i in range(R)])
        right = max([matrix[i][-1] for i in range(R)])

        ans = -1
        while left <= right:
            mid = (left + right) >> 1
            count = 0

            for i in range(R):
                count += bisect_left(matrix[i], mid)

            if count <= desired_count:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1

        return ans


# Question 44: Koko Eating Bananas
# Link: https://leetcode.com/problems/koko-eating-bananas/
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def isHourEligible(mid):
            sum_ = 0
            for i in piles:
                sum_ += (math.ceil(i/mid))
            return sum_ <= h

        left = 0
        right = max(piles)
        ans = -1
        while left <= right:
            mid = (right+left)//2

            if mid == 0:
                break

            if isHourEligible(mid):
                ans = mid
                right = mid-1
            else:
                left = mid+1
        return right if ans == -1 else ans


'''
    M = max(piles)
    Time Complexity: O(NlogM)
    Space Complexity: O(1)

'''

# Question 45: Allocate minimum number of pages
# Link: https://practice.geeksforgeeks.org/problems/allocate-minimum-number-of-pages0937/1


class Solution:
    def findPages(self, A, N, M):

        if M > N:
            return -1

        def isValid(mid):
            student = 1
            cur_pages = 0
            for i in range(N):
                cur_pages += A[i]
                if cur_pages > mid:
                    student += 1
                    cur_pages = A[i]

                if student > M:
                    return False
            return True

        left = max(A)
        right = sum(A)

        result = -1

        while left <= right:
            mid = (left + right) >> 1

            if isValid(mid):
                result = mid
                right = mid - 1
            else:
                left = mid + 1

        return result


'''
    Time Complexity: O(NlogN)
    Space Complexity: O(1)
'''

# Question 46: Longest Common Prefix
# Link: https://leetcode.com/problems/longest-common-prefix/


class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        mini = len(min(strs, key=len))
        index = 0
        answer = ""
        while index < mini:
            isEqual = True
            for i in range(len(strs)-1):
                if strs[i][index] != strs[i+1][index]:
                    isEqual = False
                    break

            if not isEqual:
                break

            answer += strs[0][index]
            index += 1
        return answer


'''
    mini = length of the smallest string
    
    Time Complexity: O(N*mini)  => O(N*N)
    Space Complexity: O(1)
'''

# Question 47: Sort Characters By Frequency
# Link: https://leetcode.com/problems/sort-characters-by-frequency/


class Solution:
    def frequencySort(self, s: str) -> str:
        lst = list(s)

        counter = Counter(lst)

        def merge(arr):
            if len(arr) <= 1:
                return

            mid = len(arr)//2

            left = arr[:mid]
            right = arr[mid:]
            merge(left)
            merge(right)

            i = j = k = 0
            while i < len(left) and j < len(right):
                if counter[left[i]] > counter[right[j]]:
                    arr[k] = left[i]
                    k += 1
                    i += 1
                elif counter[left[i]] < counter[right[j]]:
                    arr[k] = right[j]
                    j += 1
                    k += 1
                else:
                    if left[i] < right[j]:
                        arr[k] = left[i]
                        k += 1
                        i += 1
                    else:
                        arr[k] = right[j]
                        j += 1
                        k += 1

            while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1

            while j < len(right):
                arr[k] = right[j]
                j += 1
                k += 1

        merge(lst)
        return "".join(lst)


'''
    N = length of the string
    Time Complexity: O(NlogN)
    Space Complexity: O(N)
'''

# Question 48: Roman Number to Integer and vice versa
# Link: https://leetcode.com/problems/roman-to-integer/


class Solution:
    def romanToInt(self, string: str) -> int:
        mapi = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
                'C': 100, 'D': 500, 'M': 1000}
        if(len(string) == 1):
            return mapi.get(string[0])

        result = i = 0
        while(i < len(string)):
            if(i+1 < len(string) and mapi.get(string[i]) < mapi.get(string[i+1])):
                result += mapi.get(string[i+1])-mapi.get(string[i])
                i += 1
            else:
                result += mapi.get(string[i])
            i += 1
        return result


'''
    
    Time Complexity: O(N)
    Space Complexity: O(1)
'''


# Question 49: Longest Palindromic Substring (Without DP)
# link: https://leetcode.com/problems/longest-palindromic-substring/
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def lengthOfPalidrome(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1

            return s[left+1: right]

        maxi = ""
        for i in range(len(s)):
            maxi = max(maxi, lengthOfPalidrome(i, i),
                       lengthOfPalidrome(i, i+1), key=len)

        return maxi


'''
    Time Complexity: O(N^2)
    Space Complexity: O(1)
'''

# Question 50: Count number of substrings
# Link: https://practice.geeksforgeeks.org/problems/count-number-of-substrings4528/1


class Solution:
    def substrCount(self, s, k):
        ''' 
            count = 5
            dicti = { a: 1

            S = 'abaaca' K = 2
            left = 5
            right = 5

            1. For k = 2
                ans = 1 + 2 + 3 + 4 + 3 + 4

            2. For k = 1
                ans = 1 + 1 + 1 + 2 + 1 + 1

            upto k (s, k) -  upto(k-1)(s,k-1)


        '''
        def countStrings(diff):

            if diff == 0:
                return 0

            left = 0
            right = 0
            count = 0
            dicti = {}
            while right < len(s):
                if s[right] in dicti:
                    dicti[s[right]] += 1
                else:
                    dicti[s[right]] = 1

                while len(dicti) > diff:
                    dicti[s[left]] -= 1
                    if dicti[s[left]] == 0:
                        dicti.pop(s[left])
                    left += 1

                count += right - left + 1
                right += 1

            return count

        return countStrings(k) - countStrings(k-1)


'''
    Time Complexity: O(N)
    Space Complexity: O(26): O(1)
'''


# Question 51: Smallest Positive missing number
# Link: https://practice.geeksforgeeks.org/problems/smallest-positive-missing-number-1587115621/1
class Solution:
    # Function to find the smallest positive number missing from the array.
    def missingNumber(self, arr, n):
        # Your code here
        index = 0
        while index < n:

            if arr[index] > 0 and arr[index] <= n and arr[index] != arr[arr[index] - 1]:
                current_index = arr[index]-1
                temp = arr[index]
                arr[index] = arr[current_index]
                arr[current_index] = temp
            else:
                index += 1

        for i in range(n):
            if arr[i] != (i+1):
                return i+1

        return n+1


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 52: Linked List Cycle II
# Link: https://leetcode.com/problems/linked-list-cycle-ii/


class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head

        while fast and fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                while head != slow:
                    head = head.next
                    slow = slow.next

                return head


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 53: Palindrome Linked List
# Link: https://leetcode.com/problems/palindrome-linked-list/


class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head
        reverse = None
        while fast and fast.next:
            fast = fast.next.next
            temp = slow
            slow = slow.next
            temp.next = reverse
            reverse = temp

        if fast:
            slow = slow.next

        while slow and reverse and slow.val == reverse.val:
            slow = slow.next
            reverse = reverse.next

        if reverse == None:
            return True
        return False


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''
# Question 54: Segregate even and odd nodes in a Link List
# Link: https://practice.geeksforgeeks.org/problems/segregate-even-and-odd-nodes-in-a-linked-list5035/1


class Solution:
    def divide(self, N, head):
        # code here
        evenSt, evenEnd = None, None
        oddSt, oddEnd = None, None
        current = head
        while current:
            value = current.data
            if value & 1:  # odd
                if oddSt is None:
                    oddSt = current
                    oddEnd = current
                    current = current.next
                    oddEnd.next = None
                else:
                    oddEnd.next = current
                    oddEnd = current
                    current = current.next
                    oddEnd.next = None
            else:
                if evenSt is None:
                    evenSt = current
                    evenEnd = current
                    current = current.next
                    evenEnd.next = None
                else:
                    evenEnd.next = current
                    evenEnd = current
                    current = current.next
                    evenEnd.next = None
        if evenSt is None:
            return oddSt
        if oddSt is None:
            return evenSt
        evenEnd.next = oddSt
        return evenSt


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 55: Remove Nth Node From End of List
# Link: https://leetcode.com/problems/remove-nth-node-from-end-of-list/


class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], N: int) -> Optional[ListNode]:
        '''
        def recursion(node):
            p3,p2 = None,None
            p1 = head

            while p1:
                p3 = p2
                p2 = p1
                p1 = p1.next
                p2.next = p3

            return p2


        head = recursion(head)
        temp = head
        prev = None
        while temp and N-1 > 0:
            prev = temp
            temp = temp.next
            N -= 1

        if prev is None:
            head = head.next
            return recursion(head)

        prev.next = temp.next
        return recursion(head)
        '''

        def recursion(current, prev, n):
            if current == None:
                return None

            recursion(current.next, current, n)
            n[0] -= 1
            if n[0] == 0:
                if prev == None:
                    current = current.next
                    return current
                prev.next = current.next
                return prev
            return current

        return recursion(head, None, [N])


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 56: Sort Linked List
# Link: https://leetcode.com/problems/sort-list/


class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        # middle node
        def findMiddleNode(head):
            slow, fast = head, head

            while fast.next and fast.next.next:
                slow, fast = slow.next, fast.next.next

            return slow

        def merge(left, right):
            dummy = ListNode(0)
            tail = dummy

            while left and right:
                if left.val <= right.val:
                    tail.next = ListNode(left.val)
                    left = left.next
                else:
                    tail.next = ListNode(right.val)
                    right = right.next
                tail = tail.next

            while left:
                tail.next = ListNode(left.val)
                tail = tail.next
                left = left.next

            while right:
                tail.next = ListNode(right.val)
                tail = tail.next
                right = right.next

            return dummy.next

        if head and head.next:
            middle = findMiddleNode(head)
            rightPart = middle.next
            middle.next = None
            return merge(self.sortList(head), self.sortList(rightPart))

        return head


'''
    Time Complexity: O(NlogN)
    Space Complexity: O(N)
'''

# Question 57: Add 1 to a number represented as linked list
# Link: https://practice.geeksforgeeks.org/problems/add-1-to-a-number-represented-as-linked-list/1


class Solution:
    def addOne(self, head):
        # Returns new head of linked List.
        def reverse(head):
            p3, p2 = None, None
            p1 = head

            while p1:
                p3 = p2
                p2 = p1
                p1 = p1.next
                p2.next = p3

            return p2

        head = reverse(head)

        carry = 0
        temp = head
        prev = None
        isFirst = True

        while temp:
            prev = temp
            sum_ = temp.data + carry + (1 if isFirst else 0)
            temp.data = sum_ % 10
            carry = sum_//10
            if carry == 0:
                break
            temp = temp.next
            isFirst = False

        if carry == 1:
            prev.next = ListNode(1)

        return reverse(head)


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 58: Add Two Numbers
# Link: https://leetcode.com/problems/add-two-numbers/


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:

        carry = 0
        head = None
        tail = None
        while l1 != None or l2 != None or carry == 1:
            current_sum = carry
            if l1:
                current_sum += l1.val
                l1 = l1.next
            if l2:
                current_sum += l2.val
                l2 = l2.next

            newNode = ListNode(current_sum % 10)
            if head is None:
                tail = newNode
                head = newNode
            else:
                tail.next = newNode
                tail = tail.next
            carry = current_sum // 10

        return head


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 59: Intersection Point in Y Shapped Linked Lists
# Link: https://leetcode.com/problems/intersection-of-two-linked-lists/


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        def getLength(head):
            count = 0
            while head != None:
                head = head.next
                count += 1
            return count
        countA = getLength(headA)  # O(m)
        countB = getLength(headB)  # O(n)

        if countA > countB:
            temp = countA - countB
            while temp:               # O(m)
                headA = headA.next
                temp -= 1
        else:
            temp = countB - countA
            while temp:               # O(n)
                headB = headB.next
                temp -= 1

        while headA != headB:
            headA = headA.next
            headB = headB.next
        return headA


'''
    # Time: O(m) + O(n) + Max(O(m,n))
    # Space: O(1)
'''
