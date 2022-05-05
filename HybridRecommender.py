from ModelRecommender import ModelRecommender
from ContentRecommender import ContentRecommender
from CollabItemRecommender import CollabItemRecommender
import pandas as pd
import numpy as np

class HybridRecommender(ModelRecommender):
    def __init__(self):
        super().__init__()

        self.cr = ContentRecommender()
        self.cbi = CollabItemRecommender()

    def recommend_SD(self, targetTitle = 'random', limit = 5):
        if(hasattr(self.cr,'movie_features') == False):
            self.cr.readData_SD()
        if(hasattr(self.cbi,'movie_features') == False):
            self.cbi.readData()

        self.model_cbi_knn = self.cbi.createModel(self.cbi.movie_features_df)
        self.model_cr_knn = self.cr.createModel(self.cr.movie_features)

        self.recommend_duo_model(limit,targetTitle=targetTitle)

    def recommend_data_SD(self, targetTitle = 'random'):
        if(hasattr(self,'movie_features') == False):
            self.__readData_SD()
        self.__createModel(self.movie_features)
        super().recommenderData(self.movie_features,self.model_knn,targetTitle)
        
    def load_recommend_SD(self,  path_cbi = 'models/SM-hybrid-cbi', path_cr = 'models/SM-hybrid-cr', targetTitle = 'random', limit = 5):
        if(hasattr(self.cr,'movie_features') == False):
            self.cr.readData_SD()

        if(hasattr(self.cbi,'movie_features_df') == False):
            self.cbi.readData()

        self.model_cbi_knn = super().openModel(path_cbi)
        self.model_cr_knn = super().openModel(path_cr)

        self.recommend_duo_model(limit,targetTitle)

    def recommend_duo_model(self, limit : int, targetTitle = 'random'):
        query_index_cbi = np.random.choice(self.cbi.movie_features_df.shape[0])
        if(targetTitle != 'random'):
            query_index_cbi = self.cbi.movie_features_df.index.get_indexer([targetTitle])

        query_index_cr = np.random.choice(self.cr.movie_features.shape[0])
        if(targetTitle != 'random'):
            query_index_cr = self.cr.movie_features.index.get_indexer([targetTitle])

        distances_cr, indices_cr = self.model_cr_knn.kneighbors(self.cr.movie_features.
                            iloc[query_index_cr,:].values.reshape(1, -1), n_neighbors = self.cr.movie_features.axes[0].size)
        
        distances_cbi, indices_cbi = self.model_cbi_knn.kneighbors(self.cbi.movie_features_df.
                            iloc[query_index_cbi,:].values.reshape(1, -1), n_neighbors = self.cbi.movie_features_df.axes[0].size)
        
        indices_cr_list = []
        distances_cr_list = []
        for i in range(len(indices_cr[0])):
            indices_cr_list.append(indices_cr[0][i])
            distances_cr_list.append(distances_cr[0][i])

        indices_cbi_list = []
        distances_cbi_list = []
        for i in range(len(indices_cbi[0])):
            indices_cbi_list.append(indices_cbi[0][i])
            distances_cbi_list.append(distances_cbi[0][i])

        cr_df = pd.DataFrame(np.stack((indices_cr_list,distances_cr_list),axis=-1),columns=['indice','distance'])
        cbi_df = pd.DataFrame(np.stack((indices_cbi_list,distances_cbi_list),axis=-1),columns=['indice','distance'])
        cr_df = cr_df[cr_df['indice'] != query_index_cr[0]]

        score_df = pd.merge(cr_df,cbi_df, how='outer', on='indice')
        score_df = score_df.fillna(1.0)

        # distance_x = cr || distance_y = cbi
        score_df['weighted_distance'] = ((score_df['distance_x']*2) + score_df['distance_y'] )/ 3
        score_df = score_df.sort_values('weighted_distance',ascending=True)

        print('Recommendations for {0}:\n'.format(self.cr.movie_features.index[query_index_cr]))
        titles = pd.DataFrame(self.cr.movie_features.index)

        res = titles.iloc[score_df['indice'].tolist()]
        result = pd.DataFrame(res)
        result = result.merge(score_df,how='inner',left_index=True,right_on='indice')
        result = result.rename(columns={'distance_x':'content_distance','distance_y':'collab_distance'})

        result = result[result['content_distance'] < 0.9]
        result = result[result['collab_distance'] < 0.9]

        print(result[:limit][['title','collab_distance','content_distance','weighted_distance']])

    def train_and_save_SD(self, path_cbi = 'models/SM-hybrid-cbi', path_cr = 'models/SM-hybrid-cr'):
        if(hasattr(self.cr,'movie_features') == False):
            self.cr.readData_SD()
        if(hasattr(self.cbi,'movie_features_df') == False):
            self.cbi.readData()

        self.cr.createModel(self.cr.movie_features)
        self.cbi.createModel(self.cbi.movie_features_df)

        super().writeModel(path_cr,self.cr.model_knn)
        super().writeModel(path_cbi,self.cbi.model_knn)
    
    
        