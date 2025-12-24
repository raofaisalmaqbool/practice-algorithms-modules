"""
Popular Searching Algorithms
Simple and well-commented implementations for learning purposes
"""


def linear_search(arr, target):
    """
    Linear Search - Search sequentially through the array
    Time Complexity: O(n)
    Space Complexity: O(1)
    Returns: Index of target if found, -1 otherwise
    """
    # Check each element one by one
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1


def binary_search(arr, target):
    """
    Binary Search - Search in sorted array by dividing in half
    Time Complexity: O(log n)
    Space Complexity: O(1)
    Note: Array must be sorted
    Returns: Index of target if found, -1 otherwise
    """
    left = 0
    right = len(arr) - 1
    
    # Repeat until search space is empty
    while left <= right:
        # Find middle element
        mid = (left + right) // 2
        
        # Check if target is at mid
        if arr[mid] == target:
            return mid
        # If target is greater, ignore left half
        elif arr[mid] < target:
            left = mid + 1
        # If target is smaller, ignore right half
        else:
            right = mid - 1
    
    return -1


def binary_search_recursive(arr, target, left=None, right=None):
    """
    Binary Search - Recursive implementation
    Time Complexity: O(log n)
    Space Complexity: O(log n) due to recursion
    """
    if left is None:
        left = 0
    if right is None:
        right = len(arr) - 1
    
    # Base case: search space is empty
    if left > right:
        return -1
    
    # Find middle element
    mid = (left + right) // 2
    
    # Check if target is at mid
    if arr[mid] == target:
        return mid
    # Search in left half
    elif arr[mid] > target:
        return binary_search_recursive(arr, target, left, mid - 1)
    # Search in right half
    else:
        return binary_search_recursive(arr, target, mid + 1, right)


def jump_search(arr, target):
    """
    Jump Search - Jump ahead by fixed steps then linear search
    Time Complexity: O(√n)
    Space Complexity: O(1)
    Note: Array must be sorted
    Returns: Index of target if found, -1 otherwise
    """
    n = len(arr)
    # Calculate jump step size
    step = int(n ** 0.5)
    prev = 0
    
    # Jump to find the block where element may be present
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(n ** 0.5)
        # If we reached end of array
        if prev >= n:
            return -1
    
    # Linear search in the identified block
    while arr[prev] < target:
        prev += 1
        # If we reached next block or end of array
        if prev == min(step, n):
            return -1
    
    # If element is found
    if arr[prev] == target:
        return prev
    
    return -1


def interpolation_search(arr, target):
    """
    Interpolation Search - Improved binary search for uniformly distributed data
    Time Complexity: O(log log n) for uniform distribution, O(n) worst case
    Space Complexity: O(1)
    Note: Array must be sorted
    Returns: Index of target if found, -1 otherwise
    """
    left = 0
    right = len(arr) - 1
    
    while left <= right and target >= arr[left] and target <= arr[right]:
        # Avoid division by zero
        if left == right:
            if arr[left] == target:
                return left
            return -1
        
        # Calculate probe position using interpolation formula
        pos = left + int(((target - arr[left]) / (arr[right] - arr[left])) * (right - left))
        
        # Target found
        if arr[pos] == target:
            return pos
        # If target is larger, search in right subarray
        elif arr[pos] < target:
            left = pos + 1
        # If target is smaller, search in left subarray
        else:
            right = pos - 1
    
    return -1


# Example usage and demonstration
def demonstrate_searching():
    """Demonstrate all searching algorithms"""
    # Sorted array for binary, jump, and interpolation search
    sorted_arr = [2, 5, 8, 12, 16, 23, 38, 45, 56, 67, 78]
    target = 23
    
    print("Array:", sorted_arr)
    print(f"Searching for: {target}\n")
    
    print(f"Linear Search: Found at index {linear_search(sorted_arr, target)}")
    print(f"Binary Search: Found at index {binary_search(sorted_arr, target)}")
    print(f"Binary Search (Recursive): Found at index {binary_search_recursive(sorted_arr, target)}")
    print(f"Jump Search: Found at index {jump_search(sorted_arr, target)}")
    print(f"Interpolation Search: Found at index {interpolation_search(sorted_arr, target)}")
    
    # Search for non-existent element
    target = 100
    print(f"\nSearching for non-existent element: {target}")
    print(f"Linear Search: {linear_search(sorted_arr, target)}")
    print(f"Binary Search: {binary_search(sorted_arr, target)}")


if __name__ == "__main__":
    demonstrate_searching()
