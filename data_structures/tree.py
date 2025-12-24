"""
Binary Tree and Binary Search Tree Implementations
Hierarchical data structures with tree-like connections
"""


class TreeNode:
    """Node in a binary tree"""
    def __init__(self, data):
        self.data = data  # Node value
        self.left = None  # Left child
        self.right = None  # Right child


class BinarySearchTree:
    """
    Binary Search Tree - Ordered binary tree
    Left subtree < Node < Right subtree
    """
    
    def __init__(self):
        self.root = None
    
    def insert(self, data):
        """Insert a new node with given data"""
        if not self.root:
            self.root = TreeNode(data)
        else:
            self._insert_recursive(self.root, data)
    
    def _insert_recursive(self, node, data):
        """Helper function for recursive insertion"""
        # Go to left subtree
        if data < node.data:
            if node.left is None:
                node.left = TreeNode(data)
            else:
                self._insert_recursive(node.left, data)
        # Go to right subtree
        else:
            if node.right is None:
                node.right = TreeNode(data)
            else:
                self._insert_recursive(node.right, data)
    
    def search(self, data):
        """Search for a node with given data"""
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node, data):
        """Helper function for recursive search"""
        # Base cases: node is None or data is found
        if node is None or node.data == data:
            return node is not None
        
        # Data is smaller, go to left subtree
        if data < node.data:
            return self._search_recursive(node.left, data)
        # Data is larger, go to right subtree
        else:
            return self._search_recursive(node.right, data)
    
    def inorder_traversal(self):
        """
        Inorder: Left -> Root -> Right
        Returns sorted sequence for BST
        """
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        """Helper for inorder traversal"""
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.data)
            self._inorder_recursive(node.right, result)
    
    def preorder_traversal(self):
        """
        Preorder: Root -> Left -> Right
        Used for creating copy of tree
        """
        result = []
        self._preorder_recursive(self.root, result)
        return result
    
    def _preorder_recursive(self, node, result):
        """Helper for preorder traversal"""
        if node:
            result.append(node.data)
            self._preorder_recursive(node.left, result)
            self._preorder_recursive(node.right, result)
    
    def postorder_traversal(self):
        """
        Postorder: Left -> Right -> Root
        Used for deleting tree
        """
        result = []
        self._postorder_recursive(self.root, result)
        return result
    
    def _postorder_recursive(self, node, result):
        """Helper for postorder traversal"""
        if node:
            self._postorder_recursive(node.left, result)
            self._postorder_recursive(node.right, result)
            result.append(node.data)
    
    def find_min(self):
        """Find minimum value (leftmost node)"""
        if not self.root:
            return None
        
        current = self.root
        while current.left:
            current = current.left
        return current.data
    
    def find_max(self):
        """Find maximum value (rightmost node)"""
        if not self.root:
            return None
        
        current = self.root
        while current.right:
            current = current.right
        return current.data
    
    def height(self):
        """Calculate height of tree"""
        return self._height_recursive(self.root)
    
    def _height_recursive(self, node):
        """Helper for calculating height"""
        if node is None:
            return -1
        
        # Height is max of left and right subtree heights + 1
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        return max(left_height, right_height) + 1
    
    def count_nodes(self):
        """Count total number of nodes"""
        return self._count_nodes_recursive(self.root)
    
    def _count_nodes_recursive(self, node):
        """Helper for counting nodes"""
        if node is None:
            return 0
        return 1 + self._count_nodes_recursive(node.left) + self._count_nodes_recursive(node.right)


def level_order_traversal(root):
    """
    Level Order Traversal (BFS) - Visit nodes level by level
    Uses queue data structure
    """
    if not root:
        return []
    
    result = []
    queue = [root]
    
    while queue:
        # Dequeue node and add to result
        node = queue.pop(0)
        result.append(node.data)
        
        # Enqueue left and right children
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result


# Example usage and demonstration
def demonstrate_bst():
    """Demonstrate binary search tree operations"""
    print("=== Binary Search Tree Demo ===")
    bst = BinarySearchTree()
    
    # Insert nodes
    values = [50, 30, 70, 20, 40, 60, 80]
    for val in values:
        bst.insert(val)
    print(f"Inserted values: {values}")
    
    # Traversals
    print(f"\nInorder (sorted): {bst.inorder_traversal()}")
    print(f"Preorder: {bst.preorder_traversal()}")
    print(f"Postorder: {bst.postorder_traversal()}")
    print(f"Level order: {level_order_traversal(bst.root)}")
    
    # Search
    print(f"\nSearch for 40: {bst.search(40)}")
    print(f"Search for 100: {bst.search(100)}")
    
    # Min and Max
    print(f"\nMinimum value: {bst.find_min()}")
    print(f"Maximum value: {bst.find_max()}")
    
    # Height and Count
    print(f"\nTree height: {bst.height()}")
    print(f"Total nodes: {bst.count_nodes()}")


if __name__ == "__main__":
    demonstrate_bst()
