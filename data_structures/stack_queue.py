"""
Stack and Queue Implementations
Linear data structures with specific access patterns
"""


class Stack:
    """
    Stack - LIFO (Last In First Out) data structure
    Operations: push, pop, peek, is_empty
    """
    
    def __init__(self):
        self.items = []  # List to store stack elements
    
    def push(self, item):
        """Add item to top of stack"""
        self.items.append(item)
    
    def pop(self):
        """Remove and return top item from stack"""
        if self.is_empty():
            return None
        return self.items.pop()
    
    def peek(self):
        """Return top item without removing it"""
        if self.is_empty():
            return None
        return self.items[-1]
    
    def is_empty(self):
        """Check if stack is empty"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items in stack"""
        return len(self.items)
    
    def display(self):
        """Display all items in stack"""
        return self.items.copy()


class Queue:
    """
    Queue - FIFO (First In First Out) data structure
    Operations: enqueue, dequeue, front, is_empty
    """
    
    def __init__(self):
        self.items = []  # List to store queue elements
    
    def enqueue(self, item):
        """Add item to rear of queue"""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return front item from queue"""
        if self.is_empty():
            return None
        return self.items.pop(0)
    
    def front(self):
        """Return front item without removing it"""
        if self.is_empty():
            return None
        return self.items[0]
    
    def is_empty(self):
        """Check if queue is empty"""
        return len(self.items) == 0
    
    def size(self):
        """Return number of items in queue"""
        return len(self.items)
    
    def display(self):
        """Display all items in queue"""
        return self.items.copy()


class CircularQueue:
    """
    Circular Queue - Fixed size queue with wrap-around
    More efficient than regular queue for fixed size
    """
    
    def __init__(self, capacity):
        self.capacity = capacity  # Maximum size
        self.queue = [None] * capacity  # Fixed size array
        self.front = -1  # Front pointer
        self.rear = -1  # Rear pointer
        self.count = 0  # Current number of elements
    
    def enqueue(self, item):
        """Add item to queue"""
        if self.is_full():
            print("Queue is full")
            return False
        
        # First element
        if self.front == -1:
            self.front = 0
        
        # Move rear pointer circularly
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.count += 1
        return True
    
    def dequeue(self):
        """Remove and return front item"""
        if self.is_empty():
            return None
        
        item = self.queue[self.front]
        
        # Only one element
        if self.front == self.rear:
            self.front = -1
            self.rear = -1
        else:
            # Move front pointer circularly
            self.front = (self.front + 1) % self.capacity
        
        self.count -= 1
        return item
    
    def is_empty(self):
        """Check if queue is empty"""
        return self.count == 0
    
    def is_full(self):
        """Check if queue is full"""
        return self.count == self.capacity
    
    def size(self):
        """Return current size"""
        return self.count
    
    def display(self):
        """Display all items in queue"""
        if self.is_empty():
            return []
        
        elements = []
        i = self.front
        while True:
            elements.append(self.queue[i])
            if i == self.rear:
                break
            i = (i + 1) % self.capacity
        
        return elements


def check_balanced_parentheses(expression):
    """
    Use stack to check if parentheses are balanced
    Example application of stack data structure
    """
    stack = Stack()
    matching = {'(': ')', '[': ']', '{': '}'}
    
    for char in expression:
        if char in matching.keys():
            # Opening bracket - push to stack
            stack.push(char)
        elif char in matching.values():
            # Closing bracket - check if matches
            if stack.is_empty():
                return False
            if matching[stack.pop()] != char:
                return False
    
    # Stack should be empty if balanced
    return stack.is_empty()


# Example usage and demonstration
def demonstrate_stack():
    """Demonstrate stack operations"""
    print("=== Stack Demo ===")
    stack = Stack()
    
    # Push elements
    stack.push(10)
    stack.push(20)
    stack.push(30)
    print(f"Stack after pushes: {stack.display()}")
    
    # Pop element
    print(f"Popped: {stack.pop()}")
    print(f"Stack after pop: {stack.display()}")
    
    # Peek
    print(f"Top element: {stack.peek()}")
    print(f"Size: {stack.size()}")
    
    # Balanced parentheses check
    expr = "{[()()]}"
    print(f"\nIs '{expr}' balanced? {check_balanced_parentheses(expr)}")


def demonstrate_queue():
    """Demonstrate queue operations"""
    print("\n=== Queue Demo ===")
    queue = Queue()
    
    # Enqueue elements
    queue.enqueue(10)
    queue.enqueue(20)
    queue.enqueue(30)
    print(f"Queue after enqueues: {queue.display()}")
    
    # Dequeue element
    print(f"Dequeued: {queue.dequeue()}")
    print(f"Queue after dequeue: {queue.display()}")
    
    # Front element
    print(f"Front element: {queue.front()}")
    print(f"Size: {queue.size()}")


def demonstrate_circular_queue():
    """Demonstrate circular queue operations"""
    print("\n=== Circular Queue Demo ===")
    cq = CircularQueue(5)
    
    # Enqueue elements
    for i in [10, 20, 30, 40]:
        cq.enqueue(i)
    print(f"Circular Queue: {cq.display()}")
    
    # Dequeue and enqueue
    print(f"Dequeued: {cq.dequeue()}")
    cq.enqueue(50)
    cq.enqueue(60)
    print(f"After operations: {cq.display()}")


if __name__ == "__main__":
    demonstrate_stack()
    demonstrate_queue()
    demonstrate_circular_queue()
