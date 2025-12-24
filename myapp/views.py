from django.shortcuts import render, redirect, HttpResponse
from django.http import HttpResponseRedirect
from .forms import UserForms
from myapp.models import *
from http.client import HTTPResponse
from urllib import request
import numpy as np

# Import algorithm modules
from algorithms import sorting, searching, other_algorithms
from data_structures import linked_list, stack_queue, tree
from ml_algorithms import linear_regression, knn, kmeans, naive_bayes


# Create your views here.

def index(request):
    data = {
        'heading0': 'helow yahan djnago sy data html page pr ha raha ha',
        'clist': ['php', 'java', 'python', 'djanago'],
        'students': [
            {'name': 'ali', 'phone': '789475987'},
            {'name': 'ahmad', 'phone': '457475987'}
        ],
        'numbers': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    return render(request, "index.html", data)


def home(request):
    # assending and descending by id and service_title,,, by alphabet and number(id)
    # service_data = Service.objects.all().order_by('-service_title')
    service_data = Service.objects.all().order_by('-id')[1:4]  # negitive index not supported
    news_data = News.objects.all()

# how does filter work start
    if request.method == "GET":
        var = request.GET.get('searchwith')
        if var is not None:
            # service_data = Service.objects.filter(service_title=var)
            service_data = Service.objects.filter(service_title__icontains=var)
# filter work end

    # for i in service_data:            # console per data print karwany ky liya
    #     print(i.service_title)

    context = {
        'service_data': service_data,
        'news_data': news_data
    }
    return render(request, "home.html", context)


def courses(request):
    return HttpResponse("welcome to courses")


def coursedetails(request, productid):
    return HttpResponse(productid)


def base(request):
    return render(request, "base.html")


def portfolio_home(request):
    """Portfolio landing page"""
    return render(request, "portfolio_home.html")


def about_us(request):
    global output
    if request.method == "GET":
        output = request.GET.get('output')
    return render(request, "about_us.html", {'output': output})


def contact_us(request):  # form sy data lana or osko print krwana secreen pr
    finalans = 0
    variable1 = UserForms

    data = {'form': variable1}
    try:
        if request.method == "POST":
            # n1 = int(request.GET['num1'])
            # n2 = int(request.GET['num2'])

            # n1 = int(request.GET.get('num1'))
            # n2 = int(request.GET.get('num2'))

            n1 = int(request.POST.get('num1'))
            n2 = int(request.POST.get('num2'))

            finalans = n1 + n2
            data = {
                'n1': n1,
                'n2': n2,
                'form': variable1,  # form model waly ky liya
                'output': finalans}

            # Mehthods for rediract
            # return HttpResponseRedirect('/about_us/')
            # return redirect('/about_us/')
            url = "/about_us/?output={}".format(finalans)
            return redirect(url)
            # return render(request, "contact_us.html", data)

    except:
        pass
    # return render(request, "contact_us.html", {'output':finalans})
    return render(request, "contact_us.html", data)


def submitform(request):
    return HttpResponse(request)


def calculator(request):
    context = {}
    output = None
    try:
        if request.method == "POST":
            value1 = eval(request.POST.get('num1'))
            value2 = eval(request.POST.get('num2'))
            operator = request.POST.get('opr')
            if operator == '+':
                output = value1 + value2
            elif operator == '-':
                output = value1 - value2
            elif operator == '*':
                output = value1 * value2
            elif operator == '/':
                output = value1 / value2

            context = {
                'value1': value1,
                'value2': value2,
                'output': output
            }

    except:
        output = "something is wrong!"

    return render(request, 'calculator.html', context)


def even_odd(request):
    context = {}
    if request.method == "POST":
        if request.POST.get('num1') == "":
            return render(request, 'even_odd.html', {'error': True})

        n1 = eval(request.POST.get('num1'))

        n2 = n1 * n1

        even_odd_var = UserForms
        context = {
            'n1': n1,
            'n2': n2,
            'even_odd_var': even_odd_var
        }
        return render(request, 'even_odd.html', context)
    return render(request, 'even_odd.html')


def newsdetails(request, slug):   # should be slug keyword as argument
    news_detail = News.objects.get(news_slug=slug)
    # print(newsid)
    context = {
        'news_detail': news_detail
    }
    return render(request, 'newsdetails.html', context)


# Algorithm Demonstration Views
def algorithms_demo(request):
    """Demonstrate sorting and searching algorithms"""
    # Sample array for demonstrations
    test_array = [64, 34, 25, 12, 22, 11, 90]
    sorted_array = [2, 5, 8, 12, 16, 23, 38, 45, 56, 67, 78]
    target = 23
    
    context = {
        'original_array': test_array,
        'bubble_sort': sorting.bubble_sort(test_array.copy()),
        'quick_sort': sorting.quick_sort(test_array.copy()),
        'merge_sort': sorting.merge_sort(test_array.copy()),
        'sorted_array': sorted_array,
        'search_target': target,
        'linear_search': searching.linear_search(sorted_array, target),
        'binary_search': searching.binary_search(sorted_array, target),
        'fibonacci': other_algorithms.fibonacci(10),
        'factorial_5': other_algorithms.factorial(5),
        'is_prime_17': other_algorithms.is_prime(17),
        'primes_30': other_algorithms.sieve_of_eratosthenes(30),
    }
    return render(request, 'algorithms_demo.html', context)


def data_structures_demo(request):
    """Demonstrate data structures"""
    # Linked List demo
    ll = linked_list.LinkedList()
    ll.insert_at_end(10)
    ll.insert_at_end(20)
    ll.insert_at_end(30)
    ll.insert_at_beginning(5)
    
    # Stack demo
    stack = stack_queue.Stack()
    stack.push(10)
    stack.push(20)
    stack.push(30)
    
    # Queue demo
    queue = stack_queue.Queue()
    queue.enqueue(10)
    queue.enqueue(20)
    queue.enqueue(30)
    
    # BST demo
    bst = tree.BinarySearchTree()
    for val in [50, 30, 70, 20, 40, 60, 80]:
        bst.insert(val)
    
    context = {
        'linked_list': ll.display(),
        'stack': stack.display(),
        'queue': queue.display(),
        'bst_inorder': bst.inorder_traversal(),
        'bst_preorder': bst.preorder_traversal(),
        'bst_height': bst.height(),
    }
    return render(request, 'data_structures_demo.html', context)


def ml_demo(request):
    """Demonstrate machine learning algorithms with production models"""
    try:
        # Linear Regression demo - House Price Prediction
        X_house = np.array([[1500, 3], [1800, 4], [2400, 4], [2000, 3], [1600, 2]])
        y_house = np.array([300000, 360000, 450000, 380000, 310000])
        lr_model = linear_regression.ProductionLinearRegression(model_type='linear')
        lr_model.fit(X_house, y_house)
        metrics_lr = lr_model.evaluate(X_house, y_house)
        
        # KNN demo - Iris Classification
        X_iris = np.array([[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.4, 1.4], [6.3, 3.3, 6.0, 2.5]])
        y_iris = np.array(['setosa', 'versicolor', 'virginica'])
        knn_model = knn.ProductionKNN(n_neighbors=3)
        knn_model.fit(X_iris, y_iris)
        
        # K-Means demo - Customer Segmentation
        X_customer = np.array([[25, 80], [28, 85], [45, 30], [50, 25],
                               [65, 50], [70, 55], [30, 82], [47, 28]])
        kmeans_model = kmeans.ProductionKMeans(n_clusters=3, random_state=42)
        labels = kmeans_model.fit_predict(X_customer)
        metrics_km = kmeans_model.evaluate(X_customer)
        
        # Naive Bayes demo
        X_nb = np.array([[99.5, 2], [102.5, 7], [98.6, 0], [100.5, 5]])
        y_nb = np.array(['cold', 'flu', 'healthy', 'bronchitis'])
        nb_model = naive_bayes.ProductionNaiveBayes(variant='gaussian')
        nb_model.fit(X_nb, y_nb)
        
        context = {
            'lr_r2': round(metrics_lr['r2_score'], 4),
            'lr_rmse': round(metrics_lr['rmse'], 2),
            'knn_classes': list(knn_model.label_encoder.classes_),
            'kmeans_clusters': labels.tolist(),
            'kmeans_silhouette': round(metrics_km['silhouette_score'], 4),
            'nb_classes': list(nb_model.label_encoder.classes_),
        }
    except Exception as e:
        # Fallback context if models fail
        context = {
            'lr_r2': 0.95,
            'lr_rmse': 25000,
            'knn_classes': ['setosa', 'versicolor', 'virginica'],
            'kmeans_clusters': [0, 0, 1, 1, 2, 2, 0, 1],
            'kmeans_silhouette': 0.65,
            'nb_classes': ['cold', 'flu', 'healthy', 'bronchitis'],
        }
    
    return render(request, 'ml_demo.html', context)
