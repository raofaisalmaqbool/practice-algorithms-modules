"""
Linked List Implementation
A linear data structure where elements are linked using pointers
"""


class Node:
    """Single node in a linked list"""
    def __init__(self, data):
        self.data = data  # Store data
        self.next = None  # Pointer to next node


class LinkedList:
    """
    Singly Linked List implementation
    Operations: insert, delete, search, display
    """
    
    def __init__(self):
        self.head = None  # First node in list
    
    def insert_at_beginning(self, data):
        """Insert new node at the beginning"""
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def insert_at_end(self, data):
        """Insert new node at the end"""
        new_node = Node(data)
        
        # If list is empty
        if not self.head:
            self.head = new_node
            return
        
        # Traverse to the last node
        current = self.head
        while current.next:
            current = current.next
        
        current.next = new_node
    
    def insert_at_position(self, data, position):
        """Insert new node at specific position"""
        if position == 0:
            self.insert_at_beginning(data)
            return
        
        new_node = Node(data)
        current = self.head
        
        # Traverse to position-1
        for i in range(position - 1):
            if not current:
                print("Position out of range")
                return
            current = current.next
        
        # Insert node
        new_node.next = current.next
        current.next = new_node
    
    def delete_node(self, key):
        """Delete first occurrence of node with given key"""
        current = self.head
        
        # If head node holds the key
        if current and current.data == key:
            self.head = current.next
            return
        
        # Search for the key
        prev = None
        while current and current.data != key:
            prev = current
            current = current.next
        
        # Key not found
        if not current:
            print(f"Key {key} not found")
            return
        
        # Unlink the node
        prev.next = current.next
    
    def search(self, key):
        """Search for a node with given key"""
        current = self.head
        position = 0
        
        while current:
            if current.data == key:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def get_length(self):
        """Get the length of linked list"""
        count = 0
        current = self.head
        
        while current:
            count += 1
            current = current.next
        
        return count
    
    def reverse(self):
        """Reverse the linked list"""
        prev = None
        current = self.head
        
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        self.head = prev
    
    def display(self):
        """Display all nodes in the list"""
        elements = []
        current = self.head
        
        while current:
            elements.append(current.data)
            current = current.next
        
        return elements


# Example usage and demonstration
def demonstrate_linked_list():
    """Demonstrate linked list operations"""
    ll = LinkedList()
    
    print("=== Linked List Demo ===")
    
    # Insert elements
    ll.insert_at_end(10)
    ll.insert_at_end(20)
    ll.insert_at_end(30)
    ll.insert_at_beginning(5)
    print(f"After insertions: {ll.display()}")
    
    # Insert at position
    ll.insert_at_position(15, 2)
    print(f"After inserting 15 at position 2: {ll.display()}")
    
    # Search
    print(f"Position of 20: {ll.search(20)}")
    
    # Delete
    ll.delete_node(20)
    print(f"After deleting 20: {ll.display()}")
    
    # Length
    print(f"Length of list: {ll.get_length()}")
    
    # Reverse
    ll.reverse()
    print(f"After reversing: {ll.display()}")


if __name__ == "__main__":
    demonstrate_linked_list()
