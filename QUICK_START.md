# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install Django==3.2.9
pip install numpy==1.21.0
pip install django-tinymce==3.4.0
```

### Step 2: Set Up Django
```bash
# Run database migrations
python3 manage.py migrate

# (Optional) Create admin user
python3 manage.py createsuperuser
```

### Step 3: Run the Application
```bash
# Start the Django development server
python3 manage.py runserver
```

Then open your browser and visit:
- **Main Portfolio Page**: http://127.0.0.1:8000/

## 📱 Available Pages

| URL | Description |
|-----|-------------|
| `/` | Beautiful portfolio landing page |
| `/algorithms/` | Sorting, searching, and math algorithms demo |
| `/data-structures/` | Linked list, stack, queue, BST demo |
| `/ml-demo/` | Machine learning algorithms demo |
| `/calculator/` | Web calculator |
| `/home/` | Original Django home page |
| `/admin/` | Django admin panel |

## 🧪 Testing Individual Modules

You can run each module independently to see console output:

```bash
# Algorithms
python3 algorithms/sorting.py
python3 algorithms/searching.py
python3 algorithms/other_algorithms.py

# Data Structures
python3 data_structures/linked_list.py
python3 data_structures/stack_queue.py
python3 data_structures/tree.py

# Machine Learning (requires NumPy)
python3 ml_algorithms/linear_regression.py
python3 ml_algorithms/knn.py
python3 ml_algorithms/kmeans.py
python3 ml_algorithms/naive_bayes.py
```

## ⚠️ Common Issues

### Issue: "No module named 'numpy'"
**Solution**: Install NumPy
```bash
pip install numpy
```

### Issue: "No module named 'django'"
**Solution**: Install Django
```bash
pip install Django==3.2.9
```

### Issue: Port 8000 already in use
**Solution**: Use a different port
```bash
python3 manage.py runserver 8080
```

### Issue: Database errors
**Solution**: Run migrations
```bash
python3 manage.py migrate
```

## 📚 What to Show in Portfolio

1. **GitHub Repository**: Push this project to GitHub
2. **Live Demo**: Show the Django web interface
3. **Code Quality**: Highlight clean, commented code
4. **Variety**: Demonstrate algorithms, data structures, and ML
5. **Documentation**: Show the comprehensive README

## 🎯 Interview Talking Points

- "Implemented 15+ algorithms from scratch with complexity analysis"
- "Built data structures including BST with all traversals"
- "Created ML algorithms using NumPy and gradient descent"
- "Developed Django web interface to demonstrate implementations"
- "All code is well-documented and portfolio-ready"

## 🌟 Next Steps

1. ✅ Install dependencies (`pip install -r requirements.txt`)
2. ✅ Run migrations (`python3 manage.py migrate`)
3. ✅ Start server (`python3 manage.py runserver`)
4. ✅ Open http://127.0.0.1:8000/
5. ✅ Explore all demo pages
6. ✅ Push to GitHub
7. ✅ Add to your resume/portfolio

## 💡 Pro Tips

- Use virtual environment: `python3 -m venv venv && source venv/bin/activate`
- Keep code updated: Regularly commit changes to Git
- Add screenshots: Take screenshots of the demo pages for your portfolio
- Share the link: Deploy to Heroku or PythonAnywhere for live demo
- Write blog post: Explain your implementations on Medium or Dev.to

---

**Ready to impress with your portfolio! 🎉**
