
int f(int n)
{
    if (n <= 1)
    {
        return 1;
    }
    return f(n - 1) + f(n - 1);
}

/*
    Time Complexity of abvoe code

As, at each call the function is making twice.
Suppose n = 4
                        f(4)
                        /   \
                       /     \
                      /       \
                   f(3)        f(3)
                  /    \        /   \
                /       \      /     \
            f(2)       f(2)   f(2)     f(2)
           /   \      /   \    /   \    /   \
          /     \    /     \  /     \  /     \
         f(1)   f(1) f(1)f(1)f(1) f(1)f(1)   f(1)


Level |   Nodes      |           Expressed as       |
   0  |   1          |                2^0           |
   1  |   2          |                2^1           |
   2  |   4          |                2^2           |

So, 2^0 + 2^1 + 2^2 + .......... + 2^n = 2^(n+1) - 1

When, this type of pattern occurs, then time complexity would be O(branches^depth)

So, here at each node number of branches = 2
and depth is n
So, time complexity would be O(2^n)

Space: As atmost n elements till 1, will be present in the stack, so the space complexity would be O(n).

*/

/*
    Question 2
*/
int prodcut(int a, int b)
{
    int sum = 0;
    for (int i = 0; i < b; i++)
    {
        sum += a;
    }
    return sum;
}

/*
    O(b) - for loop  will run b times
*/

/*
    Question 3
*/

int power(int a, int b)
{
    if (b == 0)
    {
        return 1;
    }
    return a * power(a, b - 1);
}

/*
    O(b) - The recursive code iterates through b calls, since it subtracts 1 at each level.
*/

/*
    Question 4
*/

int mod(int a, int b)
{
    if (b <= 0)
    {
        return -1;
    }
    int div = a / b;
    return a - div * b;
}

/*
    O(1) - The code runs in constant time.
*/

/*
    Question 5
*/
int div(int a, int b)
{
    int count = 0;
    int sum = b;
    while (sum <= a)
    {
        sum += b;
        count++;
    }
    return count;
}
/*
    As, when b > a, the code runs in constant time
    When, b < a, the code runs until sum not becomes equal to a.
    So, it makes b equal to a, so the time complexity would be O(a/b)
*/

/*
    Question 6
*/

int sqrt(int n)
{
    return sqrt_helper(n, 1, n);
}

int sqrt_helper(int n, int min, int max)
{
    if (max < min)
    {
        return -1;
    }
    int guess = (min + max) / 2;
    if (guess * guess == n)
    {
        return guess;
    }
    else if (guess * guess < n)
    {
        return sqrt_helper(n, guess + 1, max);
    }
    else
    {
        return sqrt_helper(n, min, guess - 1);
    }
}
/*
    O(logn) - The algo is eventually doing binary  search, so the time complexity would be O(logn)
*/

/*
    Question 7
*/

int sqrt(int n)
{
    for (int guess = 1; guess * guess <= n; guess++)
    {
        if (guess * guess == n)
        {
            return guess;
        }
    }
    return -1;
}

/*
    O(sqrt(n)) - The algo is iterating through the loop till guess * guess <= n
*/

/*
    Question 8: If a binary tree is not balanced, how long might it take (worst case) to find an element in it?
    Answer:  O(n)
*/

/*
    Question 9: You are looking for a specific value in a binary tree, but the tree is not a binary search tree. What is the time complexity of this?
    Answer: O(n)
*/

/*
    Question 10

    int[] copyArray(int[] array){
        int[] copy = new int[0];
        for(int value: array){
            copy = appendToNew(copy, value);
        }
        return copy;
    }

    int[] appendToNew(int[] array, int value){
        int[] bigger = new int[array.length + 1];
        for(int i = 0; i < array.length; i++){
            bigger[i] = array[i];
        }
        bigger[bigger.length - 1] = value;
        return bigger;
    }


    first Time: Copy length = 1
    second Time: Copy length = 2
    third Time: Copy length = 3


    nth time: Copy length = n

    So, 1 + 2 + 3 + 4 + 5+ ... +  n = n(n+1)/2

    So, the time complexity would be O(n^2)
*/

/*

    Question 11

*/

int sumDigits(int n)
{
    int sum = 0;
    while (n > 0)
    {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

/*
    O(logn) - The runtime will be the number of digits in n. A number with d digits has log10(n) + 1 digits. So, the time complexity would be O(logn)

*/

/*
    Question 12

    int intersection(int a[], int b[]){
        mergeSort(b);
        int intersection = 0;

        for(int i = 0; i < a.length; i++){
            if(binarySearch(b, a[i]) >= 0){
                intersection++;
            }
        }
        return intersection;
    }

    time: O(blogb + alogb)
        First we have to sort the array b, so the time complexity would be O(blogb)
        Then we have to iterate through the array a, so the time complexity would be O(alogb)
*/

/*

    Question 13

    int numChars = 26;

    void printSortedStrings(int remaining){
        printSortedStrings(remaining, "");
    }

    void printSortedStrings(int remaining, String prefix){
        if(remaining == 0){
            if(isInOrder(prefix)){
                System.out.println(prefix);
            }
        } else {
            for(int i = 0; i < numChars; i++){
                char c = ithLetter(i);
                printSortedStrings(remaining - 1, prefix + c);
            }
        }
    }

    boolean isInOrder(String s){
        for(int i = 1; i < s.length(); i++){
            int prev = ithLetter(s.charAt(i - 1));
            int curr = ithLetter(s.charAt(i));
            if(prev > curr){
                return false;
            }
        }
        return true;
    }

    char ithLetter(int i){
        return (char) (((int) 'a') + i);
    }


    O(k*c^k) - Where k is the length of the string and c is the number of characters in the alphabet. It takes O(c^k) time to generate each string and O(k) time to check if it's sorted.

*/
