# k-nearest-neighbors

Naive implementation of KNN algorithm.

Example of code:

```python
clf = KNNClassifier(2)
# Training feature vectors
x = [
    [1, 2, 3, 1, 2, 1],
    [1, 4, 3, 1, 2, 3],
    [5, 1, 3, 3, 2, 1],
    [1, 2, 3, 1, 3, 1],
    [1, 2, 3, 1, 2, 3],
    [1, 2, 5, 1, 5, 1],
    [5, 2, 8, 5, 2, 5],
    [5, 4, 8, 5, 2, 8],
    [5, 5, 8, 8, 2, 5],
    [5, 2, 8, 5, 8, 5],
    [5, 2, 8, 5, 2, 8],
    [5, 2, 5, 5, 5, 5],
]
# Training labels
y = [
    "A",
    "A",
    "A",
    "A",
    "A",
    "A",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
]
clf.fit(x, y)  # "Learning" step
print(clf.predict([1, 13, 3, 1, 20, 1]))  # Prediction
```

Output:

```
A
```
