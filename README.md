# Practice Modules - Algorithm & ML Portfolio

A comprehensive Django-based portfolio project demonstrating popular algorithms, data structures, and machine learning implementations. This project is designed for learning and portfolio demonstration purposes with clean, well-commented code.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modules Overview](#modules-overview)
- [Running Examples](#running-examples)
- [Technologies Used](#technologies-used)

## ✨ Features

### 🔍 Algorithms
- **Sorting Algorithms**: Bubble Sort, Selection Sort, Insertion Sort, Merge Sort, Quick Sort
- **Searching Algorithms**: Linear Search, Binary Search, Jump Search, Interpolation Search
- **Other Algorithms**: Fibonacci, Factorial, Prime Numbers, GCD/LCM, Two Sum, Palindrome Check

### 📊 Data Structures
- **Linked List**: Singly linked list with insert, delete, search, reverse operations
- **Stack**: LIFO implementation with practical examples (balanced parentheses)
- **Queue**: FIFO implementation including circular queue
- **Binary Search Tree**: BST with traversals (inorder, preorder, postorder, level-order)

### 🤖 Machine Learning Algorithms (Production-Ready with scikit-learn)
- **Linear Regression**: Industry-standard implementation with Ridge/Lasso variants
- **K-Nearest Neighbors (KNN)**: Classification with hyperparameter tuning
- **K-Means Clustering**: Unsupervised learning with optimal k selection
- **Naive Bayes**: Probabilistic classifier with multiple variants (Gaussian, Multinomial, Bernoulli)

### 🌐 Django Web Application
- Calculator functionality
- Even/Odd checker
- News management system
- Form handling demonstrations

## 📁 Project Structure

```
practice-modules/
├── algorithms/              # Algorithm implementations
│   ├── sorting.py          # Sorting algorithms
│   ├── searching.py        # Searching algorithms
│   └── other_algorithms.py # Miscellaneous algorithms
├── data_structures/         # Data structure implementations
│   ├── linked_list.py      # Linked list
│   ├── stack_queue.py      # Stack and Queue
│   └── tree.py             # Binary Search Tree
├── ml_algorithms/          # Machine learning algorithms
│   ├── linear_regression.py # Linear regression
│   ├── knn.py              # K-Nearest Neighbors
│   ├── kmeans.py           # K-Means clustering
│   └── naive_bayes.py      # Naive Bayes classifier
├── myapp/                  # Django application
│   ├── models.py           # Database models
│   ├── views.py            # View functions
│   ├── forms.py            # Form definitions
│   └── admin.py            # Admin configuration
├── self_practice/          # Django project settings
│   ├── settings.py         # Project settings
│   ├── urls.py             # URL routing
│   └── wsgi.py             # WSGI configuration
├── templates/              # HTML templates
├── manage.py               # Django management script
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd practice-modules
   ```

2. **Create and activate virtual environment**
   ```bash
   # On Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**
   ```bash
   python manage.py migrate
   ```

5. **Create superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

6. **Run the development server**
   ```bash
   python manage.py runserver
   ```

7. **Access the application**
   - Open browser and navigate to: `http://127.0.0.1:8000/`
   - Admin panel: `http://127.0.0.1:8000/admin/`

## 💻 Usage

### Running Algorithm Examples

Each module can be run independently to see demonstrations:

```bash
# Sorting algorithms
python algorithms/sorting.py

# Searching algorithms
python algorithms/searching.py

# Other algorithms
python algorithms/other_algorithms.py

# Linked list
python data_structures/linked_list.py

# Stack and Queue
python data_structures/stack_queue.py

# Binary Search Tree
python data_structures/tree.py

# Linear Regression
python ml_algorithms/linear_regression.py

# K-Nearest Neighbors
python ml_algorithms/knn.py

# K-Means Clustering
python ml_algorithms/kmeans.py

# Naive Bayes
python ml_algorithms/naive_bayes.py
```

## 📚 Modules Overview

### Algorithms Module

#### Sorting (`algorithms/sorting.py`)
- **Bubble Sort**: Simple comparison-based sorting - O(n²)
- **Selection Sort**: Finds minimum and places at beginning - O(n²)
- **Insertion Sort**: Builds sorted array incrementally - O(n²)
- **Merge Sort**: Divide and conquer approach - O(n log n)
- **Quick Sort**: Efficient partitioning algorithm - O(n log n)

#### Searching (`algorithms/searching.py`)
- **Linear Search**: Sequential search - O(n)
- **Binary Search**: Divide and conquer on sorted data - O(log n)
- **Jump Search**: Block-based search - O(√n)
- **Interpolation Search**: Position-based search - O(log log n)

#### Other Algorithms (`algorithms/other_algorithms.py`)
- Fibonacci sequence generation
- Factorial calculation
- Prime number checking
- Sieve of Eratosthenes
- GCD and LCM
- String operations
- Two Sum problem

### Data Structures Module

#### Linked List (`data_structures/linked_list.py`)
- Insert at beginning, end, or position
- Delete by value
- Search for elements
- Reverse the list
- Get length

#### Stack and Queue (`data_structures/stack_queue.py`)
- Stack: LIFO operations (push, pop, peek)
- Queue: FIFO operations (enqueue, dequeue, front)
- Circular Queue: Fixed-size efficient queue
- Practical example: Balanced parentheses checker

#### Binary Search Tree (`data_structures/tree.py`)
- Insert and search operations
- Tree traversals (inorder, preorder, postorder, level-order)
- Find minimum and maximum values
- Calculate tree height
- Count nodes

### Machine Learning Module

#### Linear Regression (`ml_algorithms/linear_regression.py`)
- Gradient descent optimization
- Cost function tracking
- R² score evaluation
- Example: Study hours vs exam scores

#### K-Nearest Neighbors (`ml_algorithms/knn.py`)
- Distance-based classification
- Majority voting mechanism
- Accuracy evaluation
- Example: Fruit classification

#### K-Means Clustering (`ml_algorithms/kmeans.py`)
- Unsupervised learning
- Iterative centroid updates
- Inertia calculation
- Example: Customer segmentation

#### Naive Bayes (`ml_algorithms/naive_bayes.py`)
- Probabilistic classification
- Gaussian probability distribution
- Prior and likelihood calculation
- Example: Spam detection

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **Django 3.2.9**: Web framework
- **scikit-learn**: Production ML library
- **NumPy & Pandas**: Data manipulation
- **Joblib**: Model persistence
- **SQLite**: Database (default Django DB)
- **HTML/CSS**: Frontend templates
- **TinyMCE**: Rich text editor

## 📖 Learning Resources

Each implementation includes:
- ✅ Clear, descriptive comments
- ✅ Time and space complexity analysis (algorithms)
- ✅ Production best practices (ML models)
- ✅ Working examples with sample data
- ✅ Simple and readable code structure
- ✅ Demonstration functions
- ✅ Model preprocessing, validation, and persistence (ML)

## 🎯 Use Cases

This project is perfect for:
- Learning fundamental algorithms and data structures
- Understanding machine learning basics
- Portfolio demonstrations
- Interview preparation
- Teaching and educational purposes
- Quick reference for algorithm implementations

## 📝 Notes

- **Algorithms & Data Structures**: Implemented from scratch for educational understanding
- **ML Models**: Use production-ready scikit-learn (industry standard)
- Code is optimized for readability and best practices
- Each module is independent and can be studied separately
- ML implementations follow real-world production workflows

## 🤝 Contributing

This is a personal learning and portfolio project. Feel free to fork and modify for your own use.

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

**Rao Faisal Maqbool**

---

**Happy Coding! 🚀**
