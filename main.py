import heapq

import math
from collections import Counter


def euclidian_distance(p1, p2):
    sum = 0
    for e1, e2 in zip(p1, p2):
        sum += (e1 - e2) ** 2
    return math.sqrt(sum)


class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.x = []
        self.y = []

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):
        distances = []
        for datapointX, datapointY in zip(self.x, self.y):
            heapq.heappush(distances, (euclidian_distance(datapointX, x), datapointY))
        k_nearest_neighbors = heapq.nsmallest(self.k, distances)
        nearest_labels = [neighbor[1] for neighbor in k_nearest_neighbors]
        return Counter(nearest_labels).most_common(1)[0][0]


if __name__ == '__main__':
    clf = KNNClassifier(2)
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
    clf.fit(x, y)
    print(clf.predict([1, 13, 3, 1, 20, 1]))
