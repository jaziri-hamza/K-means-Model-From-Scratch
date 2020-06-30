import numpy as np
from random import randrange

"""
    Kmeans is a unsupervised learning model
    ....
    Attribute
    ---------
        X: input data
        K: number of classes
"""
class KMeans:

    def __init__(self, X, k):
        self._X = X
        if(k<=1):
            raise Exception("minimun cluster  is 2")
        self._k = k
        self._points = []
        self._points_index = []

    def _getZeros(self):
        return np.zeros( (self._X.shape[1]) )

    def fit(self, iter):
        for k in range(self._k):
            point, index = self._getRandomPoint(self._points_index)
            self._points_index.append( index )
            self._points.append(point)
        
        self._initArray()
        for i in range(iter):
            self._calcDistanceBetweenPoints()
            self._updatePoints()
            self._initArray()


    """
        update point with new cord 
        after calulate the mean of approach points of each point
    """
    def _updatePoints(self):
        for i in range(len(self._points)): # init point to zero
            self._points[i] = self._getZeros()

        point_count = np.zeros((1, len(self._points)))
        for row in range(self._X.shape[0]): # loop on all simples
            index = 0 # init index to 0
            
            for res_index in range(1, len(self._results)):
                if self._results[res_index][row] > self._results[res_index-1][row]:
                    # print( self._results[res_index][row], " -> ", self._results[res_index][row-1])
                    index = res_index
            # print(index)
            self._points[index] += self._X[row]
            point_count[0,index] +=1

        for i in range(len(self._points)):
            if( not point_count[0,i] == 0):
                self._points[i] = self._points[i] / point_count[0,i]

    
        
    """
        init the array to save the distance of each points
        results shape: (numberOfSimples, numberOfCLasses)
    """
    def _initArray(self):
        self._results = []
        for k in range(len(self._points)):
            self._results.append( np.zeros(( self._X.shape[0], 1 )) )

    """
        calculer distance between all points
    """
    def _calcDistanceBetweenPoints(self):
        for point_index in range(len(self._points)):
            for row in range( self._X.shape[0] ):
                res = 0
                for col in range( self._X.shape[1] ):
                    # print(self._points[point_index].shape)
                    res += np.square(self._points[point_index][col] - self._X[row, col])
                self._results[point_index][row,0] = np.sqrt(res)


    

    """
        unexpectedNumber: list of numbers exist
            for avoid same points 
    """
    def _getRandomPoint(self, unexpectedNumber):
        index = randrange( self._X.shape[0] )
        if(index in unexpectedNumber):
            return self._getRandomPoint(unexpectedNumber)
        return self._X[index,:], index