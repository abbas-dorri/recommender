import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle

class ModelRecommender:
    def recommend_test(self,rowPath : str,model_knn : NearestNeighbors,limit : int):
        l1 = pd.read_csv(rowPath)
        distances, indices = model_knn.kneighbors(l1.values, n_neighbors = (limit + 1))
        
        for i in range(0, len(distances.flatten())):
            if i == 0:
    
                print('Recommendations for {0}:\n'.format("Toy Story 2"))
            else:
                print('{0}: film: {1}, with distance of {2}:'.format(i, indices.flatten()[i], distances.flatten()[i]))
    
    def recommenderData(self,movie_features_df : pd.DataFrame, model_knn : NearestNeighbors, targetTitle = 'random'):
        query_index = np.random.choice(movie_features_df.shape[0])
        if(targetTitle != 'random'):
            query_index = movie_features_df.index.get_indexer([targetTitle])

        distances, indices = model_knn.kneighbors(movie_features_df.
                            iloc[query_index,:].values.reshape(1, -1))
        print(indices,query_index)
        return distances,indices
        
    def recommend(self,movie_features_df : pd.DataFrame, model_knn : NearestNeighbors,limit : int, targetTitle = 'random'):
        query_index = np.random.choice(movie_features_df.shape[0])
        if(targetTitle != 'random'):
            query_index = movie_features_df.index.get_indexer([targetTitle])

        # l1 = pd.DataFrame(movie_features_df.iloc[query_index,:].values)
        # l1.to_csv('toystory2RowContent-small.csv')

        distances, indices = model_knn.kneighbors(movie_features_df.
                            iloc[query_index,:].values.reshape(1, -1), n_neighbors = (limit + 1))
        
        for i in range(0, len(distances.flatten())):
            if i == 0:
                print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
            else:
                print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))
    
    def recommenderData(self,movie_features_df : pd.DataFrame, model_knn : NearestNeighbors, targetTitle = 'random'):
        query_index = np.random.choice(movie_features_df.shape[0])
        if(targetTitle != 'random'):
            query_index = movie_features_df.index.get_indexer([targetTitle])

        distances, indices = model_knn.kneighbors(movie_features_df.
                            iloc[query_index,:].values.reshape(1, -1))
        print(indices,query_index)
        return distances,indices
        
    def writeModel(self, path : str, model : NearestNeighbors):
        # Its important to use binary mode 
        knnPickle = open(path, 'wb')
        # source, destination 
        pickle.dump(model, knnPickle)

    def openModel(self, path: str):
        return pickle.load(open(path, 'rb'))