from bisect import bisect_right
import copy
from functools import lru_cache
import heapq
from itertools import accumulate
from optparse import Option
from tokenize import String
from typing import List, Optional
from collections import Counter, defaultdict, deque
import re
from data_structures.Trie import Trie
from data_structures.UnionFind import UnionFind


class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = self.right = None


# Question 1
class Solution:
    def tree2str(self, root: Optional[TreeNode]) -> str:
        if root is None:
            return ''

        currentValueString = str(root.val)
        leftString = ''
        if root.left:
            leftString += '(' + self.tree2str(root.left) + ')'
        elif root.right:
            leftString += '()'

        rightString = ''
        if root.right:
            rightString += '(' + self.tree2str(root.right) + ')'

        return currentValueString + leftString + rightString


'''
    * The number of nodes in the tree is in the range [1, 10^4].
    --- Time complexity ---

    Since, number of branches at each node are not fixed, It can be either 0, 1 or 2.
    In worst case, it could be 1 at each node, and tree is flat towards right as follows

            1
             \
              \
               2
                \
                 \
                  3
                   \
                    \
                     4
    In the above case, Tree travesel will be the number of nodes present in tree, It could be N.
    So, O(N).
    Now, At each node we are doing string concatenation of left and right part.
    Suppose, at root node (In the above case), we are doing concatenation of all N nodes, So, Concatenation will take O(N) time.


'''

# Question 2


class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        nearestLeafCellDepth = 0

        if root is None:
            return nearestLeafCellDepth

        queue = deque()
        queue.append(root)

        while True:
            for i in range(len(queue)):
                node = queue.popleft()
                if node.left is None and node.right is None:
                    return nearestLeafCellDepth + 1

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            nearestLeafCellDepth += 1


'''
    Time Complexity : As Similar as above.  The above solution is using level order traversel, not using dfs traversel.
                      dfs takes more time, as it traverse all nodes in average case.
                      But, in worst case they, both traverse all nodes.

        Average case:
                             1
                              \
                               \
                                2
                               / \
                              /   \
                             3     4
                            /
                           /
                          5

        Since, the nearest node is present at level 3, but dfs will do its scanning at all level, and at the end result will be returned, but in case of bfs, it will stop at level 3, as we will easily check that node 4 is leaf node, and we can stop the bfs traversel.

        Worst case:
                           1
                            \
                             \
                              2
                               \
                                \
                                 3
                                  \
                                   \
                                    4

            Both traversel will stop at level 4 only.

    Space: O(level_at_which_maximum_number_of_nodes_are_present)

'''


# Question 3
'''
Two Formats are allowed
1. (xxx) xxx-xxxx
2. xxx-xxx-xxxx

for last seven characters the regex is: [0-9]{3}\-)[0-9]{3}\-[0-9]{4}

for first four it is: (\([0-9]{3}\) |[0-9]{3}\-)
    for (xxx) :  \([0-9]{3}\)    (a Space is also present at the last)
    for xxx-  :  [0-9]{3}\-
'''


class Solution:
    def __init__(self):
        self.REGEX = r"^(\([0-9]{3}\) |[0-9]{3}\-)[0-9]{3}\-[0-9]{4}$"

    def validPhoneNumber(self, phoneNumbers=["987-123-4567", "123-456-7890", "(123) 456-7890"]):
        for number in phoneNumbers:
            if re.match(self.REGEX, number):
                print(number)


'''

    Time-Complexity:  As List Of phoneNumbers could be of length n, and each string present in the list could of length s
    So, matching each string with regex takes O(10) time, as as per REGEX, string should be only of length 10, so It will ignore other strings whose length is above 10.
    So, Time: O(10*n) = O(n)

    Space: O(1)  // No space only space of self.REGEX (Constanct sapce)

'''

# Question 4


class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        words = s.split()
        if len(words) != len(pattern):
            return False

        commonWords = set()
        commonCharacter = {}

        for index in range(len(pattern)):

            if pattern[index] in commonCharacter:
                if commonCharacter[pattern[index]] != words[index]:
                    return False
            else:
                if words[index] in commonWords:
                    return False
                commonWords.add(words[index])
                commonCharacter[pattern[index]] = words[index]

        return True


'''
    pattern length: p, string array length: a
    As, we split words by seperation of space, so maximum length could be equal to pattern length, otherwise answer is automatically false.
    Time  Complexity: O(p)  // p is length of pattern p
    At line 172, and 177: we are comparing strings. So, comparison time could be equal to string length. O(s) // s is length of string present in array s.

    So, Time Complexity: O(p*s)

    Space Complexity: O(p) // p is length of pattern p
'''


# Question 5
class Solution:
    def reverseVowels(self, s: str) -> str:
        VOWELS = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']

        s = list(s)

        left = 0
        right = len(s) - 1

        def isCharacterVowel(ch):
            return ch in VOWELS

        while left <= right:
            isLeftVowel = isCharacterVowel(s[left])
            isRightVowel = isCharacterVowel(s[right])

            isBothVowel = isLeftVowel and isRightVowel
            isBothNotVowel = (not isLeftVowel and not isRightVowel)
            if isBothVowel:
                s[left], s[right] = s[right], s[left]

            if isBothVowel or isBothNotVowel:
                left += 1
                right -= 1
            elif isLeftVowel:
                right -= 1
            else:
                left += 1

        return "".join(s)


'''
    Time complexity: At line 200, we are creating a list of string, so it takes O(n) time.
    From line 208 to 223, we are doing swapping of characters, so it takes O(n/2) time, as left<=right. so till half length of string, we will do swapping.
    At last 225, we are joining the list of string, so it takes O(n) time.
    So, Time Complexity: O(n)

    Space Complexity: O(n) // As we are creating a list of string, so it takes O(n) space.
                VOWELS: O(10)
            So, total: O(n+10) => O(n)
'''


# Question 6

class Solution:
    def inorderTraversel(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        inorder = []

        stack = [(root, False)]

        while stack:
            node, visited = stack.pop()
            if not node:
                continue
            if visited:
                inorder.append(node)
            else:
                stack.append([node.right, False])
                stack.append([node, True])
                stack.append([node.left, False])
        return inorder


'''
    * The number of nodes in the tree is in the range [1, 10^4].
    --- Time complexity ---

    Since, number of branches at each node are not fixed, It can be either 0, 1 or 2.
    In worst case, it could be 1 at each node, and tree is flat towards right as follows

            1
             \
              \
               2
                \
                 \
                  3
                   \
                    \
                     4
    In the above case, Tree travesel will be the number of nodes present in tree, It could be N.
    So, O(N).
    Now, At each node we are doing string concatenation of left and right part.
    Suppose, at root node (In the above case), we are doing concatenation of all N nodes, So, Concatenation will take O(N) time.

'''

# Question 7


class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[int]:
        result = []
        for h in range(0, 12):
            for m in range(0, 60):
                if bin(h).count('1') + bin(m).count('1') == turnedOn:
                    result.append('{:d}:{:02d}'.format(h, m))

        return result


'''
    Time Complexity: O(1) // As, we are iterating over 12 hours and 60 minutes, so it takes O(1) time.
    O(60*12) = O(1)
    line 297 is also doing constant amount of time, as numbers will go upto 6th bit.

    Space Complexity: O(1) // As, we are creating a list of string, so it takes O(1) space.
'''

# Question 8


class Solution:
    def longestPalindrome(self, s: str) -> int:
        count = Counter(s)

        evens = 0
        odds = 0
        isOddExist = False

        for ch in count:
            if count[ch] & 1 == 0:
                evens += count[ch]
            else:
                odds += count[ch] - 1
                isOddExist = True

        return evens + ((odds + 1) if isOddExist else 0)


'''

    Time Complexity: O(n) // As, we are iterating over string, so it takes O(n) time.

    Space Complexity: O(52) = O(1) // As, we are creating a list of string, so it takes O(1) space.
    AS in the string 52 different characters can be present.

'''


# Question 9

class Solution:
    def thirdMax(self, nums: List[int]) -> int:

        keys = Counter(nums).keys()

        firstMaximum = max(keys)

        if len(keys) < 3:
            return firstMaximum

        secondMaximum = float('-inf')
        thirdMaximum = float('-inf')

        for num in keys:
            if num != firstMaximum:
                secondMaximum = max(secondMaximum, num)

        for num in keys:
            if num != firstMaximum and num != secondMaximum:
                thirdMaximum = max(thirdMaximum, num)

        return thirdMaximum


'''

    Time Complexity: O(n) // As, we are iterating over keys array, and it may be possible all the keys are distinct, so it takes O(n) time.

    Space Complexity: O(n) // As, we are creating a list of string, so it takes O(n) space.


    To Reduce the above task's space complexity, we can do as follows:

        max1 = max2 = max3 = float('-inf')
        for number in nums:
            if number>max1 and number>max2 and number>max3:
                max1,max2,max3 = number,max1,max2
            elif number != max1 and number>max2 and number>max3:
                max2,max3 = number,max2
            elif number != max1 and number != max2 and number>max3:
                max3 = number

        if max3 == float('-inf'): return max1
        return max3

        The Above code has O(1) space complexity as we are using only three variables.
'''

# Question 10


class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        properties.sort(key=lambda x: (x[0], -x[1]))

        stack = []
        for a, b in properties:

            while stack and b > stack[-1]:
                stack.pop()

            stack.append(b)

        return len(properties) - len(stack)


'''
    Time Complexity: O(nlogn) // As, we are sorting the array, so it takes O(nlogn) time.

    Then, for next time, suppose, properties length is 10^5.
    All elements are having value [1, 1], [1, 2], ..... [1, 10^5-1]
    And last 10^5th element has value [2, 10^5] which is greater than all other previous.

    In this, case when we reach Nth element, we will have to compare the Nth element with all other n-1 elements, so it takes O(n) time.

    So, total time complexity: O(nlogn + n) = O(nlogn)

'''


class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()

        childIndex = cookieIndex = 0
        while childIndex < len(g) and cookieIndex < len(s):
            if g[childIndex] <= s[cookieIndex]:
                childIndex += 1
            cookieIndex += 1
        return childIndex


'''
    time for g: O(glogg)
    time for s: O(slogs)
    time for while loop: O(min(len(g),len(s)))
    total time: O(glogg + slogs + min(len(g),len(s)))

    Time: O(max(glogg, slogs)) // As, we are sorting the array, so it takes O(nlogn) time.

'''

# Question 12


class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:

        # Approach 1
        l = len(s)
        for i in range(1, l//2+1):
            if l % i == 0 and s[:i]*(l//i) == s:
                return True
        return False

        # Approach 2
        # return s in (s+s)[1:-1]


'''
    Time Complexity For Approach 1: We are iterating till half length of the string, and each time comparing it with sring s.
    So, O(n*sqrtn).

    Time complexity For Approach 2: We are doing concatenation of string s with itself, and then slicing it.
    So, O(n).

    Space Complexity: O(n) // As, we are not using any extra space.
'''

# Question 13


class Solution:
    def checkDistances(self, s: str, distance: List[int]) -> bool:
        d = defaultdict(list)

        for i, ch in enumerate(s):
            d[ch].append(i)

        return all(b-a-1 == distance[ord(ch)-97] for ch, (a, b) in d.items())


'''
    Time Complexity: O(n) // As, we are iterating over string, so it takes O(n) time.

    Space Complexity: O(26) = O(1) // As, we are creating a list of string, so it takes O(1) space.
'''

# Question 14


class Solution:
    def findSubarrays(self, nums: List[int]) -> bool:
        sums = set()

        for i in range(len(nums)-1):
            sum_ = nums[i] + nums[i+1]
            if sum_ in sums:
                return True
            sums.add(sum_)
        return False


'''

    Time Complexity of Question 14:

    length of nums could be N
    Time Complexity of for loop: O(N)

    Time Complexity to check if number in sum_ or not: O(1)
    Time Complexity for inserting an element into set: O(1)

    So, total time complexity: O(N)

'''

# Question 15


class Solution:
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        nums = list(accumulate(sorted(nums)))

        return [bisect_right(nums, q) for q in queries]


'''
    Time Complexity: O(nlogn) // As, we are sorting the array, so it takes O(nlogn) time.

    Space Complexity: O(n) // As, we are creating a list of string, so it takes O(n) space.

'''

# Question 16


class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:

        @lru_cache(None)
        def dp(i, trans, isSell):
            # base case
            if i == len(prices) or trans == 0:
                return 0

            donothing = dp(i+1, trans, isSell)

            doSomething = (prices[i] + dp(i+1, trans-1, 0)
                           ) if isSell == 1 else (-prices[i]+dp(i+1, trans, 1))
            return max(donothing, doSomething)

        return dp(0, k, 0)


'''
    Time Complexity:
                    p = length of prices array
                    k = number of transactions to be permitted.

                O(p*k)

    space Complexity: O(p*k) as cache will store maximum p*k states.

'''

# Question 17


class Solution:
    def minimumRecolors(self, blocks: str, k: int) -> int:
        '''
            Check in each k size window of given string, how many whites needed to be colored into white.
        '''

        n, mini = len(blocks), float('inf')
        for i in range(n-k+1):
            white = blocks.count('W', i, i + k)
            mini = min(mini, white)
        return mini


'''
    Time Comlexity: iterating till n-k+1, so O(n-k+1) = O(n).
    Inside the for loop, each time, we are calculating White count in each k size window of string. So, O(k).

    So, total time complexity: O(n*k)

    Space Complexity: O(1) // As we are not using any extra space.
'''

# Question 18


class Solution:
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        MOD = 10**9+7

        ZIPPED = sorted(zip(efficiency, speed), reverse=True)
        ANS = 0
        SPEED = 0

        HEAP = []
        for E, S in ZIPPED:

            SPEED += S
            heapq.heappush(HEAP, S)

            if len(HEAP) > k:
                SPEED -= heapq.heappop(HEAP)

            ANS = max(ANS, E*SPEED)

        return ANS % MOD


'''
    Time Complexity: O(nlogn) // As, we are sorting the array, so it takes O(nlogn) time.
    Time Complexity of for loop: O(n) and inside for loop we are maintaining a heap, so it takes O(logk) time.

    So, total time complexity of for loop: O(nlogk)
    So, Actual time complexity: O(nlogn + nlogk) =
    Space Complexity: O(n) // As, we are creating a list of string, so it takes O(n) space.

'''


# Question 19
class Solution:
    def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
        tokens.sort()
        left = 0
        right = len(tokens)-1
        score = 0

        while left <= right:
            if power >= tokens[left]:
                power -= tokens[left]
                score += 1
                left += 1
            elif score > 0 and left != right:
                power += tokens[right]
                right -= 1
                score -= 1
            else:
                break

        return score


'''
    Time Complexity: O(nlogn) // As, we are sorting the array, so it takes O(nlogn) time.

    while loop time complexity, Suppose, power >= sum(all_tokens)
    Then, pointer will move 1 by 1 towards left, so complete array will be scanned, so takes O(n).

    O(nlogn) + O(n) = O(nlogn)

    Space Complexity: O(1) // As, we are not using any extra space.

'''

# Question 20


class Solution:
    def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
        rows, cols = len(grid), len(grid[0])
        resultedMatrix = [[0]*(cols-2) for _ in range(rows-2)]

        for i in range(rows-2):
            for j in range(cols-2):
                maxi = 0
                for ii in range(i, i+3):
                    for jj in range(j, j+3):
                        maxi = max(maxi, grid[ii][jj])
                resultedMatrix[i][j] = maxi
        return resultedMatrix


'''
Time Complexity:
    At line 675: we are iterating O((rows-2)*(cols-2)) times.
    from line 677, we are iterating O((rows-2)*(cols-2)) times.
    Inside the for loop from 677, we are againg iterating a 3*3 matrix each time, so O(9) = O(1) times.
    So, total time complexity: O((rows-2)*(cols-2)) + O((rows-2)*(cols-2)) + O(1) = O((rows-2)*(cols-2))

Space Complexity: O((rows-2)*(cols-2)) // As, we are creating a list of string, so it takes O((rows-2)*(cols-2)) space.

'''

# Question 21


class Solution:
    def minNumberOfHours(self, initialEnergy: int, initialExperience: int, energy: List[int], experience: List[int]) -> int:
        hours = 0
        for i in range(len(energy)):
            if initialEnergy <= energy[i]:
                hours += energy[i] - initialEnergy + 1
                initialEnergy += energy[i] - initialEnergy + 1

            if initialExperience <= experience[i]:
                hours += experience[i] - initialExperience + 1
                initialExperience += experience[i] - initialExperience + 1

            initialEnergy -= energy[i]
            initialExperience += experience[i]
        return hours


'''

    Time Complexity: O(n) // As, we are iterating the array, so it takes O(n) time.

    Space Complexity: O(1) // As, we are not using any extra space.

'''


# Question 22
class Solution:
    def repeatedCharacter(self, s: str) -> str:
        occurence = [0]*26
        for ch in s:
            index = ord(ch) - ord('a')
            occurence[index] += 1
            if occurence[index] >= 2:
                return ch


'''
    String length could be s, we are iterating the string, so it takes O(s) time.

    Space Complexity: O(1) // As, we are not using any extra space.

'''

# Question 23


class Solution:
    def numberOfPairs(self, nums: List[int]) -> List[int]:
        count = Counter(nums)
        result = [0, 0]
        for key in count:
            a, b = divmod(count[key], 2)
            result[0] += a
            result[1] += b
        return result


'''
    Time Complexity: O(n) // As, we are iterating the array, so it takes O(n) time.

    Space Complexity: O(n) // As, we are creating a dictionary, so it takes O(n) space.

'''

# Question 24


class Solution:
    def fillCups(self, amount: List[int]) -> int:
        heap = []

        seconds = 0
        for i in range(3):
            if amount[i] != 0:
                heapq.heappush(heap, -amount[i])

        while len(heap) >= 2:
            firstMax, secondMax = -heapq.heappop(heap), -heapq.heappop(heap)

            seconds += 1

            firstMax -= 1
            secondMax -= 1
            if firstMax > 0:
                heapq.heappush(heap, -firstMax)
            if secondMax > 0:
                heapq.heappush(heap, -secondMax)

        if len(heap) == 1:
            number = -heapq.heappop(heap)
            seconds += number
        return seconds

        # another Solution
        # return max(max(amount), (sum(amount) + 1)//2)   # Efficient Solution


'''
    The loop will run for maximum of max(amount) times, so it takes O(max(amount)) time.
    Space Complexity: O(1) // As, we are not using any extra space, just using a heap of size 3, which is constant.

    # Time Complexity for another solution: O(3) // As, we are iterating the array for sum, so it takes O(n) time.
    # Space Complexity for another solution: O(1)

'''


# Question 25
'''
UTF-8 Rules
1. A character can be from 1bytes to 4 bytes.
2. If a character is from 1 byte, then it's first bit is 0.
3. If a character is from 2 bytes, then it's first 2 bits are 110.
4. If a character is from 3 bytes, then it's first 3 bits are 1110.
5. If a character is from 4 bytes, then it's first 4 bits are 11110.

    Number of Bytes       UTF-8 Encoding
        1 byte               0xxxxxxx
        2 bytes              110xxxxx 10xxxxxx
        3 bytes              1110xxxx 10xxxxxx 10xxxxxx
        4 bytes              11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

    Ex: [197, 130, 1]
    Binary form: [11000101, 10000010, 00000001]

    The first two numbers are belongs to 2 bytes, as 110xxxxx, 10xxxxxx
    The third number is belongs to 1 byte, as 0xxxxxxx

    So, It is a valid sequence.


'''


class Solution:
    def validUtf8(self, data: List[int]) -> bool:

        remainigBytes = 0

        for number in data:
            if remainigBytes == 0:
                if number >> 7 == 0:
                    remainigBytes = 0
                elif number >> 5 == 6:
                    remainigBytes = 1
                elif number >> 4 == 14:
                    remainigBytes = 2
                elif number >> 3 == 30:
                    remainigBytes = 3
                else:
                    return False
            else:
                if number >> 6 != 2:
                    return False
                remainigBytes -= 1
        return remainigBytes == 0


'''
    Time Complexity: O(n) // As, we are iterating the array, so it takes O(n) time.
    In while loop, for each number we are moving upto 8 bits towards left, which is not a big task. O(1)
    So, total time complexity: O(n) + O(1) = O(n)

    Space Complexity: O(1) // As, we are not using any extra space.
'''


# Question 26

class Solution:
    def decodeMessage(self, key: str, message: str) -> str:
        sub = dict()
        for k in key:
            if k.islower() and k not in sub:
                sub[k] = chr(ord('a')+len(sub))
        sub[' '] = ' '

        return ''.join(sub[m] for m in message)


'''
    Time Complexity
    At first loop, we are iterating len(key) times.
    At second loop, we are iterating len(message) times.

    So, total time complexity: O(len(key)) + O(len(message)) = O(len(key) + len(message))

    Space Complexity: We are using a dictionary which contains 26 unique lowercase english letters mapping with their value.
    So, it takes O(26) = O(1) space.
'''

# Question 27


class Solution:
    def countAsterisks(self, s: str) -> int:

        stringsAfterSplit = s.split("|")

        return sum([string.count("*") for string in stringsAfterSplit[::2]])


'''

    Time Complexity
    1. Spliting the string, suppose (In worst case) there are bars after each character, It means if string length is s, then s/2 bars are present in the string.
    2. So, when split, maximum length of stringsAfterSplit is   s/2 + 1
    3. Now, we are iterating the stringsAfterSplit, so it takes O(s/2 + 1) time.
      In each iteration, we are iterating splitted string, but sum of length of all splitted string can not be above len(s/2)
    4. So, total time complexity: O(s/2 + 1) + O(s/2) = O(s/2 + 1) + O(s/2) = O(s)

    Space Complexity: O(s/2+1) // As, we are creating a list of size s/2+1, so it takes O(s/2+1) space.

'''

# Question 28


class Solution:
    def minMaxGame(self, nums: List[int], length=1) -> int:
        if len(nums) == length:
            return nums[0]

        index = 0
        isMini = True
        for i in range(0, len(nums)//length, 2):

            if isMini:
                nums[index] = min(nums[i], nums[i+1])
            else:
                nums[index] = max(nums[i], nums[i+1])

            index += 1
            isMini = not isMini
        return self.minMaxGame(nums, length*2)


'''
    At each step, we are iterating length/1 array, initially length is n, so will iterate for n times, but skipping one number each time. So, O(n/2) is the number of iterations.

    At each iteration, we are comparing two numbers, so it takes O(1) time.

    At 1 level:   n//2
    At 2 level:   n//4
    At 3 level:   n//8
    At 4 level:   n//16

    So, total time complexity: O(n/2) + O(n/4) + O(n/8) + O(n/16) + ... + O(1) = O(n)

    Maximum number of levels are:  O(log2(len(nums)))
    Since maximum length of nums 1024, so maximum number of levels are 10.



    Space Complexity: O(1) // As, we are not using any extra space, updating the given array nums.


'''


# Question 29
class Solution:
    def rearrangeCharacters(self, s: str, target: str) -> int:
        counterS = Counter(s)
        counterTarget = Counter(target)

        mini = float('inf')

        for key in counterTarget:
            # If a character of target is not present in s, then we can not rearrange it.
            if key not in counterS:
                return 0
            else:  # If a character of target is present in s, then we can rearrange it.
                if counterS[key] < counterTarget[key]:
                    # If number of occurences of a character in s is less than target, then we can not rearrange it.
                    return 0
                # If number of occurences of a character in s is greater than target, then we can rearrange it.
                mini = min(mini, counterS[key]//counterTarget[key])

        return mini


'''
    Time Complexity
    len(s) = s, len(target) = t
    1. counting characters in s and target. O(s+t)
    2. Iterating counterTarget, O(t)
    So, total time complexity: O(s+t) + O(t) = O(s+t)

    Space Complexity: O(s+t) // As, we are creating two dictionaries of size s and t.

'''

# Question 30


class Solution:
    def digitCount(self, num: str) -> bool:
        c = Counter(map(int, num))
        return all(c[i] == int(d) for i, d in enumerate(num))


'''
    Time Complexity
    1. Converting string to list of integers, O(n)
    2. Creating a dictionary, O(n)
    3. Iterating the given num string, O(n)
    4. Comparing the dictionary value with the number at index, O(n)

    So, total time complexity: O(n) + O(n) + O(n) + O(n) = O(n)

    Space Complexity: O(n) // As, we are creating a dictionary of size n.

'''


# Question 31
class Solution:
    def largestGoodInteger(self, num: str) -> str:
        res = ''
        cnt = 1
        for i in range(1, len(num)):
            if num[i] == num[i-1]:
                cnt += 1
            else:
                cnt = 1
            if cnt == 3:
                res = max(res, num[i] * 3)

        return res


'''
    Time Complexity
    1. Iterating the given num string, O(n)
    So, total time complexity: O(n)

    Space Complexity: O(1) // As, we are not using any extra space.

'''


# Question 32
class Solution:
    def countPrefixes(self, words: List[str], s: str) -> int:
        return sum([s.startswith(word) for word in words])

# Trie Solution of above question 32


class Solution:
    def countPrefixes(self, words: List[str], s: str) -> int:
        trie = Trie()
        for word in words:
            trie.insert(word)

        return trie.countPrefixesInString(s)


'''

    Time Complexity
    1. Iterating the given words list, O(len(words))

    Space Complexity: O(len(words)) // As, we are not using any extra space.

'''


# Question 33
class Solution:
    def pseudoPalindromicPaths(self, root: Optional[TreeNode]) -> int:
        self.ans = 0

        def dfs(node, mask):
            if not node:
                return

            # lst.append(node.val)
            mask ^= (1 << node.val)
            if not node.left and not node.right:
                # flag = 0
                # for key,value in Counter(lst).items():
                #     if value&1 == 1:
                #         flag += 1
                # # print(lst,flag)
                # if flag == 0 or flag == 1:
                #     self.ans += 1
                if mask & (mask-1) == 0:
                    self.ans += 1

                return

            dfs(node.left, mask)
            dfs(node.right, mask)

        dfs(root, 0)
        return self.ans


'''

    Time Complexity
    1. Iterating the given tree, O(n)

    Space Complexity: O(1) // As, we are not using any extra space.

'''

# Question 34


class Solution:
    def intersection(self, nums: List[List[int]]) -> List[int]:
        return sorted([k for k, v in Counter([el for lst in nums for el in lst]).items() if v == len(nums)])


'''
    Time Complexity
    Since we are iterating nums 2-d array, so O(rows*cols)
    Since we are iterating the dictionary, so O(rows*cols)

    Space Complexity: O(rows*cols) // As, we are creating a dictionary of size rows*cols.

'''


# Question 35
class Solution:
    def digitSum(self, s: str, k: int) -> str:
        if len(s) <= k:
            return s

        string = ""
        temp = 0
        count = 0
        for i in range(0, len(s)):
            temp += int(s[i])
            count += 1

            if count == k:
                string += str(temp)
                count = 0
                temp = 0

        if count != 0:
            string += str(temp)
        return self.digitSum(string, k)


'''

    Time Complexity
    1. Iterating the given string, O(n)

    Space Complexity: O(logk(n))

'''


# Question 36
class Solution:
    def findClosestNumber(self, A: List[int]) -> int:
        return max([-abs(a), a] for a in A)[1]


'''
    Time Complexity
    1. Iterating the given array, O(n)

    Space Complexity: O(n)

'''


# Question 37
class Solution:
    def checkTree(self, root: Optional[TreeNode]) -> bool:
        return root.val == (root.left.val + root.right.val)


'''
    Time Complexity. Since root, root.left and root.right exist.
    So, Time: O(1)

    Space Complexity: O(1) // As, we are not using any extra space.

'''

# Question 38


class Solution:
    def sum(self, num1: int, num2: int) -> int:
        return num1+num2


'''
    Time Complexity: O(1)
    Space Complexity: O(1)
'''


# Question 39
class Solution:
    def largestInteger(self, num: int) -> int:
        '''
            even = [6, 8]  -> [8, 6]

            odd = [5,7,5]  -> [7,5,5]

        '''

        even = []
        odd = []
        arr = []

        while num > 0:
            num, mod = divmod(num, 10)
            if mod & 1:
                odd.append(mod)
            else:
                even.append(mod)
            arr.append(mod)

        even.sort(reverse=True)
        odd.sort(reverse=True)

        number = 0
        evenIndex = 0
        oddIndex = 0

        for i in range(len(arr)):
            if arr[len(arr) - i - 1] & 1 == 0:
                number = number*10 + even[evenIndex]
                evenIndex += 1
            else:
                number = number*10 + odd[oddIndex]
                oddIndex += 1

        return number


# Question 40
class Solution:
    def findOriginalArray(self, changed: List[int]) -> List[int]:
        counter = Counter(changed)
        res = []
        for k in counter.keys():

            if k == 0:
                # handle zero as special case
                if counter[k] % 2 > 0:
                    return []
                res += [0] * (counter[k] // 2)

            elif counter[k] > 0:
                x = k

                # walk down the chain
                while x % 2 == 0 and x // 2 in counter:
                    x = x // 2

                # walk up and process all numbers within the chain. mark the counts as 0
                while x in counter:
                    if counter[x] > 0:
                        res += [x] * counter[x]
                        if counter[x+x] < counter[x]:
                            return []
                        counter[x+x] -= counter[x]
                        counter[x] = 0
                    x += x
        return res

# Question 41


class Solution:
    def maximumScore(self, nums: List[int], mult: List[int]) -> int:
        n = len(nums)
        m = len(mult)

#         @lru_cache(1000)
#         def recur(index,left):
#             if index == m:
#                 return 0

#             # if (index,left) in memo: return memo[(index,left)]

#             currMult = mult[index]
#             right = n - 1 - (index - left)

#             return max(currMult * nums[left] + recur(index+1,left+1), currMult*nums[right] + recur(index+1,left))
#             # return memo[(index,left)]

#         # memo = {}
#         return recur(0,0)

        dp = [[0]*(m+1) for _ in range(m+1)]

        for i in range(m-1, -1, -1):
            for left in range(i, -1, -1):
                currMult = mult[i]
                right = n - 1 - (i-left)
                dp[i][left] = max(currMult*nums[left] + dp[i+1]
                                  [left+1], currMult*nums[right] + dp[i+1][left])
        return dp[0][0]


# Question 42
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        if not nums1 or not nums2:
            return 0

#         @lru_cache(None)
#         def recursion(i,j,count):
#             if i<0 or j<0:
#                 return count

#             count2 = count

#             if nums1[i] == nums2[j]:
#                 count2 = recursion(i-1,j-1,count+1)

#             return max(count2, max(recursion(i-1,j, 0), recursion(i,j-1,0)))

#         return recursion(len(nums1)-1, len(nums2) - 1, 0)


#         dp = [[0]*(len(nums2) + 1) for _ in range(len(nums1) + 1)]

#         maxi = 0
#         for i in range(1, len(nums1) + 1):
#             for j in range(1, len(nums2) + 1):
#                 if nums1[i-1] == nums2[j-1]:
#                     dp[i][j] = dp[i-1][j-1] + 1
#                     maxi = max(maxi, dp[i][j])
#                 else:
#                     dp[i][j] = 0

#         return maxi

        first = ''.join(map(chr, nums1))
        second = ''.join(map(chr, nums2))

        def check(mid):

            seen = {first[i:i+mid] for i in range(len(first)-mid+1)}
            return any(second[i:i+mid] in seen for i in range(len(second)-mid+1))

        low = 0
        high = min(len(nums1), len(nums2)) + 1

        result = 0
        while low < high:
            mid = high - (high-low)//2
            if check(mid):
                low = mid
                result = max(result, low)
            else:
                high = mid-1

        return result


'''
    m= len(nums1)
    n = len(nums2) 
    Time Complexity of  first two solution is: O(m*n)

    Time Complexity of third solution is as follows: 
        since we are doing binary search over the 0 to min(len(nums1), len(nums2))
        So, log(min(m,n)))
        and for each iteration we are doing a check which is O(m+n)

        So, O(log(min(m, n))*(m+n))

    Space Complexity of first two solution is: O(m+n)
    
'''

# Question 43


class Solution:
    def smallestEvenMultiple(self, n: int) -> int:
        def lcm(x, y):
            greater = max(x, y)
            while True:
                if greater % x == 0 and greater % y == 0:
                    return greater

                greater += 1

        return lcm(n, 2)


'''
    Time Complexity of lcm of two numbers is O(max(x,y))


'''

# Question 44


class Solution:
    def licenseKeyFormatting(self, S: str, K: int) -> str:
        S = S.replace("-", "").upper()[::-1]
        return '-'.join(S[i:i+K] for i in range(0, len(S), K))[::-1]


'''
    Time Complexity: O(n)
    Space Complexity: O(n)
'''

# Question 45


class Solution:
    def convertTime(self, current: str, correct: str) -> int:
        current = 60*int(current[:2]) + int(current[3:5])
        correct = 60*int(correct[:2]) + int(correct[3:5])
        diff = correct-current
        operations = 0
        for time in [60, 15, 5, 1]:
            operations += (diff//time)
            diff = diff % time

        return operations


'''
    Time Complexity: O(1)

    Space Complexity: O(1)
'''


# Question 46
class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        initialEvenSum = sum([el for el in nums if el % 2 == 0])

        result = []
        for value, index in queries:
            if nums[index] % 2 == 0:
                initialEvenSum -= nums[index]

            nums[index] += value

            if nums[index] % 2 == 0:
                initialEvenSum += nums[index]

            result.append(initialEvenSum)
        return result


'''
    n = len(nums), q = len(queries)

    Time Complexity: O(n+q)
    Space Complexity: O(q) // result array length.
'''


# Question 47
class Solution:
    def countDaysTogether(self, aA: str, la: str, aB: str, lB: str) -> int:
        months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        def countDays(days):
            return int(days[3:]) + sum(months[:int(days[:2])-1])

        return max(0, countDays(min(la, lB)) - countDays(max(aA, aB)) + 1)


'''
    Time Complexity: O(1)
    Space Complexity: O(1)

'''

# Question 48


class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        result = 0

        while start > 0 or goal > 0:
            if start & 1 != goal & 1:
                result += 1

            start >>= 1
            goal >>= 1

        return result


'''
    Time Complexity: O(32)  // maxium 32-bit number
    Space Complexity: O(1)
'''

# Question 49


class Solution:
    def countHillValley(self, nums: List[int]) -> int:
        hillValley = 0

        for i in range(1, len(nums)-1):
            if nums[i] == nums[i+1]:
                nums[i] = nums[i-1]

            if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                hillValley += 1

            if nums[i] < nums[i-1] and nums[i] < nums[i+1]:
                hillValley += 1

        return hillValley


'''
    Time Complexity: O(n)
    Space Complexity: O(1)

'''

# Question 50


class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        count = Counter(nums)
        for key, val in count.items():
            if val & 1 == 1:
                return False
        return True


'''
    n = len(nums)
    Time Complexity: O(n)
    Space Complexity: O(n)
'''

# Question 51


class Solution:
    def cellsInRange(self, s: str) -> List[str]:
        c1, r1, _, c2, r2 = map(ord, s)
        return [chr(c) + chr(r) for c in range(c1, c2+1) for r in range(r1, r2+1)]


'''
    Time Complexity: O(1) // since there are at most 26 * 9 cells.
    Space Complexity: O(1) 
'''

# Question 52


class Solution:
    def mostFrequent(self, nums: List[int], key: int) -> int:
        c = Counter()
        for i, n in enumerate(nums):
            if n == key and i + 1 < len(nums):
                c[nums[i + 1]] += 1
        return c.most_common(1)[0][0]


'''
    N = len(nums)
    Time Complexity: O(N)
    Space: O(R) where `R` is the range of numbers in `nums`. We can also make it the count of unique numbers in `nums` if we use `hash_map`.

'''

# Question 53


class Solution:
    def prefixCount(self, words: List[str], pref: str) -> int:
        return sum([1 for word in words if word.startswith(pref)])


'''
    N = len(words), W = maximumWordlength in words.

    Time Complexity: O(NW)
    Space Complexity: O(N)
'''

# Question 54


class Solution:
    def reverseWords(self, s: str) -> str:
        r = 0
        l = 0
        while r < len(s):

            while r < len(s) and s[r] != " ":
                r += 1

            # Each time concatenate the entire srting.
            s = s[:l] + s[l:r][::-1] + s[r:]

            r += 1
            l = r
        return s


'''
    N = len(s)
    # Each time concatenate the entire srting.
    # So, concatenation takes O(N) time.

    Time Complexity: O(N^2)
    Space Complexity: O(1)
'''

# Question 55


class Solution:
    def countEven(self, num: int) -> int:

        def digitSum(num):
            sum_ = 0
            while num > 0:
                num, mod = divmod(num, 10)
                sum_ += mod
            return sum_

        return (num - digitSum(num) % 2) // 2


'''
    Time Complexity: O(logN), where base is 10  
    Space Complexity: O(1)
'''

# Question 56


class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        hashmap = defaultdict(list)
        result = 0
        for i in range(len(nums)):
            if nums[i] in hashmap:
                for x in hashmap[nums[i]]:
                    if (i*x) % k == 0:
                        result += 1

            hashmap[nums[i]].append(i)
        return result


'''
    In case of all numbers are same, then Time will O(N^2).

    Time Complexity: O(N^2)
    Space Complexity: O(N)
'''

# Question 57


class Solution:
    def countOperations(self, num1: int, num2: int) -> int:
        result = 0
        while min(num1, num2) > 0:
            if num1 < num2:
                num1, num2 = num2, num1
            ops, num1 = divmod(num1, num2)
            result += ops
        return result


'''

    Time Complexity: O(log(min(num1, num2)))
    Space Complexity: O(1)
'''


# Question 58
class Solution:
    def sortEvenOdd(self, nums: List[int]) -> List[int]:
        nums[::2], nums[1::2] = sorted(
            nums[::2]), sorted(nums[1::2], reverse=True)
        return nums


'''
    Time Complexity: O(NlogN)
    Space Complexity: O(1)

'''

# Question 59


class Solution:
    def minimumSum(self, num: int) -> int:
        num = sorted(str(num), reverse=True)
        return int(num[0]) + int(num[1]) + int(num[2])*10 + int(num[3])*10


'''
    Time Complexity: O(1)
    Space Complexity: O(1)
'''

# Question 60


class Solution:
    def concatenatedBinary(self, n: int) -> int:
        string = ""
        for i in range(1, n+1):
            string += bin(i)[2:]
        return int(string, 2) % (10**9+7)


'''
    Time Complexity: O(N)
    Space Complexity: O(N*32)
'''

# Question 61


class Solution:
    def findFinalValue(self, nums: List[int], original: int) -> int:
        count = Counter(nums)
        while original in count:
            original *= 2

        return original


'''
    Time Complexity: O(N)
    Space Complexity: O(R) // where R is the range of numbers in nums.
'''

# Question 62


class Solution:
    def countElements(self, nums: List[int]) -> int:

        return max(0, len(nums) - nums.count(min(nums)) - nums.count(max(nums)))


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 63


class Solution:
    def minimumCost(self, cost: List[int]) -> int:
        return sum(cost) - sum(sorted(cost)[-3::-3])


'''
    N = len(cost)
    Time Complexity: O(NlogN)
    Space Complexity: O(1)
'''

# Question 64


class Solution:
    def divideString(self, s: str, k: int, fill: str) -> List[str]:
        l = []
        if len(s) % k != 0:
            s += fill*(k-len(s) % k)
        for i in range(0, len(s), k):
            l.append(s[i:i+k])
        return l


'''
    N = len(s)  
    Time Complexity: O(N+k)
    Space Complexity: O(N+k)

'''

# Question 65


class Solution:
    def checkValid(self, matrix: List[List[int]]) -> bool:
        n = len(matrix)
        for r in range(n):
            row = bytearray(n + 1)
            col = bytearray(n + 1)
            print(row)
            for c in range(n):
                ro, co = matrix[r][c], matrix[c][r]
                row[ro] += 1
                col[co] += 1
                if row[ro] > 1 or col[co] > 1:
                    return False
        return True


'''
    N = row = column = len(matrix)
    NxN matrix is given
    Time Complexity: O(N^2)
    Space Complexity: O(N)

'''

# Question 66


class Solution:
    def capitalizeTitle(self, title: str) -> str:
        ans = []
        for s in title.split():
            if len(s) < 3:
                ans.append(s.lower())
            else:
                ans.append(s.capitalize())
        return ' '.join(ans)


'''
    N = len(title)
    k = number_of_spaces_in_title

    Suppose Each string length after splitting is s
    Time Complexity: O((k+1)*s)

    Space Complexity: O(k+1)

'''

# Question 67


class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        self.ans = []

        def recur(node, path, tempSum):
            if node is None:
                return 0

            if node.left == None and node.right == None:
                tempSum += node.val
                path.append(node.val)
                if tempSum == targetSum:
                    self.ans.append(copy.deepcopy(path))
                    tempSum -= node.val
                    path.pop()
                    return
            else:
                tempSum += node.val
                path.append(node.val)

            recur(node.left, path, tempSum)
            recur(node.right, path, tempSum)
            path.pop()
        recur(root, [], 0)
        return self.ans


'''
    N is the number of nodes in the tree.

    Time Complexity: O(N)
    Space Complexity: O(N)
'''

# Question 68


class Solution:
    def checkString(self, s: str) -> bool:
        return "ba" not in s


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 69
# Solutions 69.1 - DFS
'''
    1. Create A Bidirection graph for equal pairs, like a-b,b-a.
    2. Now, check for each unequal pair, whether it any unequal pair are reachable from each other, if yes, return False
    3. If above condition fails, then return True.
    
'''


class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:

        def is_node_target_reachable_from_a(a, target, visited):
            if a == target:
                return True

            visited.add(a)
            for neighbour in graph[a]:
                if neighbour not in visited and is_node_target_reachable_from_a(neighbour, target, visited):
                    return True

            return False

        graph = defaultdict(list)
        not_equals_nodes = []
        for eq in equations:
            left, sign, right = eq[0], eq[1]+eq[2], eq[3]
            if sign == "==":
                graph[left].append(right)
                graph[right].append(left)
            else:
                not_equals_nodes.append((left, right))

        for left, right in not_equals_nodes:
            if is_node_target_reachable_from_a(left, right, set()):
                return False
        return True


'''
    Time Complexity: O(N)
    Space Complexity: O(N)
'''
# Question 69
# Solutions 69.2 - Union-Find
'''
    1. Group all equal pairs.
    2. Now, iterate for each unequal pair, whether any pair exist in equal group union, If yes, Then Return False
    3. At the end, if above condition fails for each unequal pair, return True.

'''


class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:

        uf = UnionFind(26)

        not_equals_nodes = []
        for eq in equations:
            left, sign, right = eq[0], eq[1]+eq[2], eq[3]

            if sign == "==":
                uf.union(ord(left) - ord('a'), ord(right) - ord('a'))
                continue

            not_equals_nodes.append((left, right))

        for left, right in not_equals_nodes:
            if uf.find(ord(left) - ord('a')) == uf.find(ord(right) - ord('a')):
                return False

        return True


'''
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 70


class Solution:
    def isSameAfterReversals(self, num: int) -> bool:

        return num < 10 or num % 10 != 0


'''
    Time Complexity: O(1)
    Space Complexity: O(1)
'''

# Question 71


class Solution:
    def mostWordsFound(self, sentences: List[str]) -> int:
        return max([sent.count(" ") + 1 for sent in sentences])


'''
    N = len(sentences)
    S = 100  ( As per the Question )
    Time Complexity: O(N*S)  // We are counting spaces inside the sentence.
    Space Complexity: O(N)

'''

# Question 72


class Solution:
    def firstPalindrome(self, words: List[str]) -> str:
        lst = [string for string in words if string == string[::-1]]
        return lst[0] if lst else ""


'''
    N = len(words)
    S = 100  ( As per the Question )
    Time Complexity: O(N*S)  // We are reversing the string, and comparing both The Strings.

    Space Complexity: O(N)
'''

# Question 73


class Solution:
    def countPoints(self, s: str) -> int:
        count = [0]*10

        for i in range(0, len(s), 2):
            color, index = s[i], s[i+1]
            if color == "R":
                color = 1
            elif color == "G":
                color = 2
            else:
                color = 4

            count[int(index)] |= color

        return count.count(7)


'''
    Time Complexity: O(N)
    Space Complexity: O(10) -> O(1)
'''

# Question 74


class Solution:
    def maxSubsequence(self, nums: List[int], k: int) -> List[int]:
        nums = sorted([(nums[i], i)
                      for i in range(len(nums))], reverse=True)[:k]

        nums = sorted(nums, key=lambda x: (x[1], x[0]))
        return [ans[0] for ans in nums]


'''
    N = len(nums)
    Time Complexity: O(NlogN)    +  O(klogk)
                // first sorting +  second sorting

    Space Complexity: O(N)

'''

# Question 75


class Solution:
    def targetIndices(self, nums: List[int], target: int) -> List[int]:
        lt_count, eq_count = 0, 0
        for n in nums:
            if n < target:
                lt_count += 1
            elif n == target:
                eq_count += 1

        return list(range(lt_count, lt_count+eq_count))


'''
    N = len(nums)
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 76


class Solution:
    def countWords(self, words1: List[str], words2: List[str]) -> int:
        c1 = Counter(words1)
        c2 = Counter(words2)
        c = 0
        for i in words1:
            if i in words2 and c1[i] == c2[i] == 1:
                c += 1
        return c


'''
    N = len(words1)
    M = len(words2)
    Time Complexity: O(N+M)
    Space Complexity: O(N+M)
'''

# Question 77


class Solution:
    def pushDominoes(self, s: str) -> str:
        right = -1
        s = list(s)
        for i in range(len(s)):
            if s[i] == "L":
                if right == -1:
                    j = i-1
                    while j >= 0 and s[j] == ".":
                        s[j] = "L"
                        j -= 1
                else:
                    j = right + 1
                    k = i - 1
                    while j < k:
                        s[j] = "R"
                        s[k] = "L"
                        j += 1
                        k -= 1
                    right = -1

            elif s[i] == "R":
                if right != -1:
                    for j in range(right+1, i):
                        s[j] = "R"
                right = i

        if right != -1:
            for j in range(right, len(s)):
                s[j] = "R"

        return "".join(s)


''' 
    N = len(s)
    Time Complexity: O(N)
    Space Complexity: O(N)
'''


# Question 78
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        lst = []
        start = head
        while start:
            lst.append(start)
            start = start.next
        if(len(lst) < n):
            return head
        if(len(lst) == n and len(lst) > 1):
            head = lst[1]
            return head
        if(len(lst) == n and len(lst) == 1):
            return None
        lst[-n-1].next = lst[-n].next
        return head


''' 
    N = Number of nodes in the Linked List
    Time Complexity: O(N)
    Space Complexity: O(N)
'''

# Question 79


class Solution:
    def findEvenNumbers(self, digits: List[int]) -> List[int]:
        counter = [0]*10
        for d in digits:
            counter[d] += 1

        result = []
        for num in range(100, 999, 2):
            # check digits in num
            num_count = [0]*10
            temp = num
            while temp:
                num_count[temp % 10] += 1
                temp //= 10
            flag = False
            for i in range(10):
                if num_count[i] > counter[i]:
                    flag = True
                    break
            if not flag:
                result.append(num)

        return result


'''
    N = len(digits)
    first loop: O(N)
    second loop: O(1000)
    Time Complexity: O(N+1000)

    Space Complexity: O(10) -> O(1)
'''

# Question 80


class Solution:
    def maxDistance(self, colors: List[int]) -> int:
        distance = 0

        for i in range(len(colors)):
            color = colors[i]
            if color != colors[0]:
                distance = max(distance, i)

            if color != colors[-1]:
                distance = max(distance, len(colors) - i - 1)

        return distance


'''
    N = len(colors)
    Time Complexity: O(N)
    Space Complexity: O(1)

'''

# Question 81

'''
    left and equalt to the kth ticket: min(A[k], A[i]) ( When i <= k )
    right to the kth ticket: max(A[k] - 1, A[i])  ( When i > k )
'''


class Solution:
    def timeRequiredToBuy(seld, tickets: List[int], k: int) -> int:
        time = 0
        for i in range(len(tickets)):
            time += min(tickets[k] - (i > k), tickets[i])

        return time


'''
    N = len(tickets)
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 82


class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        distanceFromX = []

        for i in range(len(arr)):
            dis = abs(arr[i] - x)
            distanceFromX.append((dis, arr[i], i))

        distanceFromX.sort()

        result = ([dis[1] for dis in distanceFromX])[:k]

        return sorted(result)


'''
    N = len(arr)
    Time Complexity: O(NlogN)
    Space Complexity: O(N)
'''

# Question 83


class Solution:
    def checkAlmostEquivalent(self, word1: str, word2: str) -> bool:
        f1, f2 = [0] * 26, [0] * 26

        for ch in word1:
            f1[ord(ch) - ord('a')] += 1

        for ch in word2:
            f2[ord(ch) - ord('a')] += 1

        for i in range(26):
            if abs(f1[i] - f2[i]) > 3:
                return False

        return True


'''
    N = len(word1)
    M = len(word2)
    Time Complexity: O(N+M)
    Space Complexity: O(26) -> O(1)
'''

# Question 84


class Solution:
    def countVowelSubstrings(self, word: str) -> int:

        def atMost(s, goal):
            ans = 0
            i = 0
            count = defaultdict(int)

            for j in range(len(s)):
                if s[j] not in ['a', 'e', 'i', 'o', 'u']:
                    i = j + 1
                    count.clear()
                    continue

                count[s[j]] += 1
                while len(count) > goal:
                    count[s[i]] -= 1
                    if count[s[i]] == 0:
                        count.pop(s[i])
                    i += 1

                ans += j - i + 1

            return ans

        return atMost(word, 5) - atMost(word, 4)


'''
    N = len(word)
    Time Complexity: O(N+N) -> O(N)
    Space Complexity: O(5) -> O(1)
'''

# Question 85


class Solution:
    def smallestEqual(self, nums: List[int]) -> int:
        return next((i for i, x in enumerate(nums) if i % 10 == x), -1)


'''
    N = len(nums)
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 86


class Solution:
    def kthDistinct(self, arr: List[str], k: int) -> str:
        freq = Counter(arr)
        for x in arr:
            if freq[x] == 1:
                k -= 1
            if k == 0:
                return x
        return ""


'''
    N = len(arr)
    Time Complexity: O(N)
    Space Complexity: O(N)
'''

# Question 87


class Solution:
    def countValidWords(self, sentence: str) -> int:
        pattern = re.compile(r'(^[a-z]+(-[a-z]+)?)?[,.!]?$')
        return sum(bool(pattern.match(word)) for word in sentence.split())


''' 
    N = len(sentence)
    Time Complexity: O(N)
    Space Complexity: O(1)
'''

# Question 88


class Solution:
    def areNumbersAscending(self, s: str) -> bool:
        nums = [int(w) for w in s.split() if w.isdigit()]
        return all(nums[i-1] < nums[i] for i in range(1, len(nums)))


'''
    N = len(s)
    Time Complexity: O(N)
    Space Complexity: O(N)
'''

# Question 89


class Solution:
    def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
        seats.sort()
        students.sort()
        return sum(abs(e - t) for e, t in zip(seats, students))


'''
    N = len(seats)
    M = len(students)
    Time Complexity: O(NlogN+MlogM)
    Space Complexity: O(1)
'''

# Question 90


class Solution:
    def twoOutOfThree(self, nums1: List[int], nums2: List[int], nums3: List[int]) -> List[int]:
        count = defaultdict(int)

        def checkNumbers(nums):
            for i in set(nums):
                count[i] += 1

        checkNumbers(nums1)
        checkNumbers(nums2)
        checkNumbers(nums3)
        return [i for i, c in count.items() if c >= 2]


'''
    N = len(nums1)
    M = len(nums2)
    P = len(nums3)
    Time Complexity: O(N+M+P)
    Space Complexity: O(N+M+P)
'''

# Question 91


class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        self.ans = False

        def recur(node, tempSum):
            if node is None:
                return 0

            if node.left == None and node.right == None:
                tempSum += node.val
                if tempSum == targetSum:
                    tempSum -= node.val
                    self.ans = True
                    return
            else:
                tempSum += node.val

            recur(node.left, tempSum)
            recur(node.right, tempSum)

        recur(root, 0)
        return self.ans


'''
    Time Complexity: O(N)
    Space Complexity: O(N)
'''

# Question 92


class Solution:
    def equalFrequency(self, word: str) -> bool:
        count = Counter(word)

        for ch in word:
            count[ch] -= 1
            if count[ch] == 0:
                count.pop(ch)

            if len(set(count.values())) == 1:
                return True

            count[ch] += 1

        return False


'''
    N = len(word)
    Time Complexity: O(N)
    Space Complexity: O(N)
'''
