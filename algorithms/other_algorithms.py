"""
Other Popular Algorithms
Simple implementations of commonly used algorithms
"""


def fibonacci(n):
    """
    Fibonacci Sequence - Each number is sum of previous two
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    # Start with first two fibonacci numbers
    fib_seq = [0, 1]
    
    # Generate remaining numbers
    for i in range(2, n):
        fib_seq.append(fib_seq[i-1] + fib_seq[i-2])
    
    return fib_seq


def fibonacci_recursive(n):
    """
    Fibonacci - Recursive implementation (less efficient)
    Time Complexity: O(2^n)
    Space Complexity: O(n)
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)


def factorial(n):
    """
    Factorial - Product of all positive integers up to n
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n < 0:
        return None
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def factorial_recursive(n):
    """
    Factorial - Recursive implementation
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if n < 0:
        return None
    if n == 0 or n == 1:
        return 1
    return n * factorial_recursive(n - 1)


def is_prime(n):
    """
    Prime Number Check - Determine if number is prime
    Time Complexity: O(√n)
    Space Complexity: O(1)
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to square root
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def sieve_of_eratosthenes(limit):
    """
    Sieve of Eratosthenes - Find all primes up to limit
    Time Complexity: O(n log log n)
    Space Complexity: O(n)
    """
    if limit < 2:
        return []
    
    # Create boolean array, initially all True
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    # Start with smallest prime number, 2
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            # Mark all multiples as not prime
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    # Collect all prime numbers
    return [i for i in range(limit + 1) if is_prime[i]]


def gcd(a, b):
    """
    Greatest Common Divisor - Euclidean algorithm
    Time Complexity: O(log min(a,b))
    Space Complexity: O(1)
    """
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """
    Least Common Multiple
    Time Complexity: O(log min(a,b))
    Space Complexity: O(1)
    """
    return abs(a * b) // gcd(a, b)


def reverse_string(s):
    """
    Reverse String - Multiple approaches
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Method 1: Using slicing (Pythonic way)
    return s[::-1]


def is_palindrome(s):
    """
    Palindrome Check - Check if string reads same forwards and backwards
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # Remove spaces and convert to lowercase
    s = s.replace(" ", "").lower()
    
    # Compare with reversed string
    return s == s[::-1]


def two_sum(nums, target):
    """
    Two Sum Problem - Find two numbers that add up to target
    Time Complexity: O(n)
    Space Complexity: O(n)
    Returns: Indices of the two numbers, or None if not found
    """
    # Dictionary to store visited numbers and their indices
    seen = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        # Check if complement exists in dictionary
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return None


def find_duplicates(arr):
    """
    Find Duplicate Elements in array
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    seen = set()
    duplicates = set()
    
    for num in arr:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)
    
    return list(duplicates)


# Example usage and demonstration
def demonstrate_algorithms():
    """Demonstrate various algorithms"""
    print("=== Fibonacci Sequence ===")
    print(f"First 10 Fibonacci numbers: {fibonacci(10)}")
    
    print("\n=== Factorial ===")
    print(f"Factorial of 5: {factorial(5)}")
    
    print("\n=== Prime Numbers ===")
    print(f"Is 17 prime? {is_prime(17)}")
    print(f"Primes up to 30: {sieve_of_eratosthenes(30)}")
    
    print("\n=== GCD and LCM ===")
    print(f"GCD of 48 and 18: {gcd(48, 18)}")
    print(f"LCM of 12 and 15: {lcm(12, 15)}")
    
    print("\n=== String Operations ===")
    print(f"Reverse of 'hello': {reverse_string('hello')}")
    print(f"Is 'racecar' a palindrome? {is_palindrome('racecar')}")
    
    print("\n=== Two Sum Problem ===")
    nums = [2, 7, 11, 15]
    target = 9
    print(f"Array: {nums}, Target: {target}")
    print(f"Indices: {two_sum(nums, target)}")
    
    print("\n=== Find Duplicates ===")
    arr = [1, 2, 3, 2, 4, 5, 3, 6]
    print(f"Array: {arr}")
    print(f"Duplicates: {find_duplicates(arr)}")


if __name__ == "__main__":
    demonstrate_algorithms()
