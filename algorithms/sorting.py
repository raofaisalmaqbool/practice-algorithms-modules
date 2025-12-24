"""
Popular Sorting Algorithms
Simple and well-commented implementations for learning purposes
"""


def bubble_sort(arr):
    """
    Bubble Sort - Compare adjacent elements and swap if needed
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    n = len(arr)
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Swap if current element is greater than next
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def selection_sort(arr):
    """
    Selection Sort - Find minimum element and place it at beginning
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    n = len(arr)
    # Traverse through all array elements
    for i in range(n):
        # Find minimum element in remaining unsorted array
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        # Swap the found minimum with the first element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def insertion_sort(arr):
    """
    Insertion Sort - Build sorted array one item at a time
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    # Traverse from 1 to len(arr)
    for i in range(1, len(arr)):
        key = arr[i]
        # Move elements greater than key one position ahead
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def merge_sort(arr):
    """
    Merge Sort - Divide and conquer algorithm
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(arr) <= 1:
        return arr
    
    # Divide the array into two halves
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Merge the sorted halves
    return merge(left, right)


def merge(left, right):
    """Helper function to merge two sorted arrays"""
    result = []
    i = j = 0
    
    # Compare elements from left and right arrays
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quick_sort(arr):
    """
    Quick Sort - Divide and conquer with pivot element
    Time Complexity: O(n log n) average, O(n²) worst
    Space Complexity: O(log n)
    """
    if len(arr) <= 1:
        return arr
    
    # Choose pivot (middle element)
    pivot = arr[len(arr) // 2]
    
    # Partition array into three parts
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    # Recursively sort and combine
    return quick_sort(left) + middle + quick_sort(right)


# Example usage and demonstration
def demonstrate_sorting():
    """Demonstrate all sorting algorithms"""
    test_arr = [64, 34, 25, 12, 22, 11, 90]
    
    print("Original array:", test_arr)
    print("\nBubble Sort:", bubble_sort(test_arr.copy()))
    print("Selection Sort:", selection_sort(test_arr.copy()))
    print("Insertion Sort:", insertion_sort(test_arr.copy()))
    print("Merge Sort:", merge_sort(test_arr.copy()))
    print("Quick Sort:", quick_sort(test_arr.copy()))


if __name__ == "__main__":
    demonstrate_sorting()
