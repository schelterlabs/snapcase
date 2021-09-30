from datasketch import MinHash as MinHashBase
from datasketch import MinHashLSH
import numpy as np
from sklearn.neighbors import DistanceMetric
from scipy.spatial import distance
class MyMinHashLSH:
    """
    A wrapper for LSH in place of KNN with the same API structure.
    """

    def __init__(self, n_neighbors=100, threshold=0.5, num_perm=128):
        """

        Args:
            n_neighbors: number of neighbors to search for
            threshold: the similarity threshold for which to return query sets
            num_perm: number of permutations for minhashing, the larger the more accurate at cost of speed
        """
        # Create LSH index
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.n_neighbors = n_neighbors
        self.train_X = []

    def _numeric_to_index(self, v):
        basket = np.argwhere(v > 0.0).flatten()
        return list(map(str, basket))

    def _numerical_to_binary(self,v):
        return [1 if x > 0.0 else 0 for x in v]

    def hash(self, v):
        """Input is a one dimensional array of numerical value, e.g., [0,0.1, 0.3, 1]"""
        minhash = MinHashBase(num_perm=self.num_perm)
        # convert numerical value to string of item ids ["1", "2", "3"]
        basket_set = self._numeric_to_index(v)
        for b in basket_set:
            minhash.update(b.encode('utf8'))
        return basket_set, minhash

    def fit(self, X):
        """
        X is n by m nd array
        """
        num_rows = len(X)
        for i in range(num_rows):
            basket_set, minhash_sig = self.hash(X[i])
            self.train_X.append(self._numerical_to_binary(X[i]))
            self.lsh.insert(str(i), minhash_sig)

    def kneighbors(self, X):
        num_rows = len(X)
        distances = []
        indices = []
        for i in range(num_rows):
            result = self.lsh.query(self.hash(X[i])[1])
            # the result is the index to the train_X
            print(f"Approximate neighbours with Jaccard similarity > {self.threshold} count", len(result))
            candidate_index = [int(r) for r in result]
            candidates =np.array([self.train_X[i] for i in candidate_index])
            input_bin = self._numerical_to_binary(X[i])
            paired_dist = distance.cdist(np.array([input_bin]), candidates, 'jaccard').flatten()
            sorted_zip = sorted(zip(paired_dist, candidate_index), reverse=False)
            sorted_ind = [x[1] for x in sorted_zip]
            sorted_dist = [x[0] for x in sorted_zip]
            distances.append(sorted_dist)
            indices.append(sorted_ind)
        return np.array(distances), np.array(indices)


if __name__ == '__main__':
    set1 = [0, 0.2, 0.3]
    set2 = [0, 0, 0.1]
    set3 = [0.2, 0.3, 0.1]
    np.random.seed(10)
    train_X = np.array([set1, set2, set3])
    train_X = np.random.rand(100,3)
    ms = MyMinHashLSH(threshold=0.4, num_perm=1280)
    ms.fit(train_X)
    dist, ind = ms.kneighbors(np.array([set1]))
    print(dist)
    print(ind)

    # print(ms.lsh.keys())

    # set1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
    #         'estimating', 'the', 'similarity', 'between', 'datasets']
    # set2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
    #         'estimating', 'the', 'similarity', 'between', 'documents']
    # set3 = ['minhash', 'is', 'probability', 'data', 'structure', 'for',
    #         'estimating', 'the', 'similarity', 'between', 'documents']

    # set2 = ["0", "0", "1"]
    # set3 = ["1", "1", "1"]

    # the datasketch lib converts it to a set internally
    # set1 = ["1", "2"]
    # set2 = ["2"]
    # set3 = ["1", "2", "3"]

    # m1 = MinHash(num_perm=1280)
    # m2 = MinHash(num_perm=1280)
    # m3 = MinHash(num_perm=1280)
    # for d in set1:
    #     m1.update(d.encode('utf8'))
    # for d in set2:
    #     m2.update(d.encode('utf8'))
    # for d in set3:
    #     m3.update(d.encode('utf8'))

    # Create LSH index
    # lsh = MinHashLSH(threshold=0.1, num_perm=1280)
    # lsh.insert("m2", m2)
    # lsh.insert("m3", m3)
    # # lsh.insert("m1", m1)
    # result = lsh.query(m1)
    # print("Approximate neighbours with Jaccard similarity", result)
