try:
    import sys
    # import bisect
    # import math
    # import string
    # import heapq
    import collections
    # i_m = 9223372036854775807
    M = 10**9+7

    # def matrix(n):
    #     return [list(map(int, input().split()))for i in range(n)]

    # def string2intlist(s):
    #     return list(map(int, s))

    # def calculate_sum(a, N):  # sum of a to N
    #     # Number of multiples
    #     m = N / a
    #     # sum of first m natural numbers
    #     sum = m * (m + 1) / 2
    #     # sum of multiples
    #     ans = a * sum
    #     return ans

    # def series(N):
    #     return (N*(N+1))//2

    # def count2Dmatrix(i, list):
    #     return sum(c.count(i) for c in list)

    # def modinv(n, p):
    #     return pow(n, p - 2, p)

    # def nCr(n, r):
    #     i = 1
    #     while i < r:
    #         n *= (n - i)
    #         i += 1
    #     return n // math.factorial(r)

    # def GCD(x, y):
    #     return math.gcd(x,y)

    # def LCM(x, y):
    #     return (x * y) // GCD(x, y)

    # def Divisors(n):
    #     l = []
    #     for i in range(1, int(math.sqrt(n) + 1)):
    #         if (n % i == 0):
    #             if (n // i == i):
    #                 l.append(i)
    #             else:
    #                 l.append(i)
    #                 l.append(n//i)
    #     return l

    # # def isprime(n):
    # #     for i in range(2, int(math.sqrt(n))+1):
    # #         if n % i == 0:
    # #             return False
    # #     return True

    # def isprime(n):
    #     if(n <= 1):
    #         return False
    #     if(n <= 3):
    #         return True
    #     if(n % 2 == 0 or n % 3 == 0):
    #         return False
    #     for i in range(5,int(math.sqrt(n) + 1), 6):
    #         if(n % i == 0 or n % (i + 2) == 0):
    #             return False
    #     return True

    # def SieveOfEratosthenes(n):
    #     prime = [True for i in range(n+1)]
    #     p = 2
    #     while (p * p <= n):
    #         if (prime[p] == True):
    #             for i in range(p * p, n+1, p):
    #                 prime[i] = False
    #         p += 1
    #     f = []
    #     for p in range(2, n+1):
    #         if prime[p]:
    #             f.append(p)
    #     return f


#     q = []


#     def dfs(n, d, v, c):
#         global q
#         v[n] = 1
#         x = d[n]
#         q.append(n)
#         j = c
#         for i in x:
#             if i not in v:
#                 f = dfs(i, d, v, c+1)
#                 j = max(j, f)
#                 # print(f)
#         return j
#     # d = {}


#     def knapSack(W, wt, val, n):
#         K = [[0 for x in range(W + 1)] for x in range(n + 1)]
#         for i in range(n + 1):
#             for w in range(W + 1):
#                 if i == 0 or w == 0:
#                     K[i][w] = 0
#                 elif wt[i-1] <= w:
#                     K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
#                 else:
#                     K[i][w] = K[i-1][w]

#         return K[n][W]

    # def modularExponentiation(x, n):
    #     if(n == 0):
    #         return 1
    #     elif (n % 2 == 0):  # n is even
    #         return modularExponentiation((x*x) % M, n//2)
    #     else:  # n is odd
    #         return (x * modularExponentiation((x * x) % M, (n - 1) // 2)) % M

    # def powerOftwo(n):
    #     return n and (not (n & (n-1)))

    # def modInverse(a, m):
    #     m0 = m
    #     y = 0
    #     x = 1

    #     if (m == 1):
    #         return 0

    #     while (a > 1):

    #         # q is quotient
    #         q = a // m

    #         t = m

    #         # m is remainder now, process
    #         # same as Euclid's algo
    #         m = a % m
    #         a = t
    #         t = y

    #         # Update x and y
    #         y = x - q * y
    #         x = t

    #     # Make x positive
    #     if (x < 0):
    #         x = x + m0

    #     return x
    # temp = [0]*101
#     def mapi(l,r):
#         idx = -1
#         val = -1
#         if(l > r):
#             return 0
#         if(l == r):
#             return l
#         for l in range(l,r+1):
#             if temp[l]>val:
#     	        idx = l
#     	        val = temp[l]
#         return idx

    # class UnionFind:
    #     def __init__(self, n):
    #         self.parent = [i for i in range(n+1)]
    #         self.rank = [0]*(n+1)

    #     def find(self, u):
    #         if self.parent[u] == u:
    #             return u
    #         return self.find(self.parent[u])

    #     def union(self, u, v):
    #         xroot = self.find(u)
    #         yroot = self.find(v)
    #         if(self.rank[xroot] < self.rank[yroot]):
    #             self.parent[xroot] = yroot
    #         elif(self.rank[xroot] > self.rank[yroot]):
    #             self.parent[yroot] = xroot
    #         else:
    #             self.parent[yroot] = xroot
    #             self.rank[xroot] += 1

    # def printList(lst, isEnd=False):
    #     print(*lst, sep="", end="\n" if isEnd else "")


    def solve():
        size = 5
        arr = [3, 4, 5, 1, 1]
        sortedarray = sorted(arr)
        index = 1
        i = 0
        while index < size:
            arr[index] = sortedarray[i]
            i += 1
            index += 2

        index = 0
        while index < size:
            arr[index] = sortedarray[i]
            i += 1
            index += 2
        presum = []
        for i in range(size):
            if i == 0:
                presum.append(arr[i])
            else:
                presum.append(presum[i-1] + arr[i])

        for i in range(size):
            if i+1 == 1:
                continue
            else:
                presum[i] = (((-1)**(i))*presum[i]) + presum[i-1]
        print(presum[-1])
    # test = int(input())
    test = 1
    count1 = 1
    while count1 <= test:
        ans = solve()
        # sys.stdout.write("Case #" + str(count1) + ":" + "\n")
        # solve()
        # sys.stdout.write("Case #" + str(count1) + ": " + str(ans) + "\n")
        # sys.stdout.write(str(ans) + "\n")
        count1 += 1

except EOFError as e:
    print("")
