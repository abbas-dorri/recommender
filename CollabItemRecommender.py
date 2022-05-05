import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from ModelRecommender import ModelRecommender

class CollabItemRecommender(ModelRecommender):
    
    def readData(self):
        movies_df=pd.read_csv('data-small/movies.csv', usecols=['movieId','title'], dtype={'movieId':'int32','title':'str'})
        ratings_df=pd.read_csv('data-small/ratings.csv', usecols=['userId', 'movieId', 'rating','timestamp'],dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        
        movies_merged_df=movies_df.merge(ratings_df, on='movieId')

        movies_rating_count=movies_merged_df.groupby('title')['rating'].count().sort_values(ascending=True).reset_index().rename(columns={'rating':'Rating Count'}) #ascending=False
        popularity_threshold = 60
        popular_movies= movies_rating_count[movies_rating_count['Rating Count']>=popularity_threshold]
        movies_merged_df = movies_merged_df[movies_merged_df['title'].isin(popular_movies['title'])]
        movie_features_df=movies_merged_df.pivot_table(index='title',columns='userId',values='rating').fillna(0)
        self.movie_features_df = movie_features_df

    def readDataLarge(self):
        print('Loading in Large dataset.... this might take a while')
        movies_df=pd.read_csv('data-big/movies.csv', usecols=['id','title'], dtype={'id':'int32','title':'str'})
        ratings_df=pd.read_csv('data-big/ratings.csv', usecols=['userId', 'movieId', 'rating'],dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
         
        movies_merged_df=movies_df.merge(ratings_df, left_on='id', right_on='movieId')

        movies_rating_count=movies_merged_df.groupby('title')['rating'].count().sort_values(ascending=True).reset_index().rename(columns={'rating':'Rating Count'}) #ascending=False
        popularity_threshold = 50
        popular_movies= movies_rating_count[movies_rating_count['Rating Count']>=popularity_threshold]
        movies_merged_df = movies_merged_df[movies_merged_df['title'].isin(popular_movies['title'])]

        movie_features_df=movies_merged_df.pivot_table(index='title',columns='userId',values='rating').fillna(0)
        self.movie_features_df_L = movie_features_df

    def createModel(self, movie_features_df : pd.DataFrame):
        movie_features_df_matrix = csr_matrix(movie_features_df.values)
        model_knn = NearestNeighbors(metric = 'cosine', algorithm='brute')
        model_knn.fit(movie_features_df_matrix)

        self.model_knn = model_knn
        return model_knn

    def recommend_LD(self, targetTitle = 'random', limit = 5):
        if(hasattr(self,'movie_features_df_L') == False):
            self.readDataLarge()
        self.createModel(self.movie_features_df_L)
        super().recommend(self.movie_features_df_L,self.model_knn,limit,targetTitle)
    
    def recommend_SD(self, targetTitle = 'random', limit = 5):
        if(hasattr(self,'movie_features_df') == False):
            self.readData()
        self.createModel(self.movie_features_df)
        super().recommend(self.movie_features_df,self.model_knn,limit,targetTitle)
        
    def recommend_data_SD(self,targetTitle = 'random'):
        if(hasattr(self,'movie_features_df') == False):
            self.readData()
        self.createModel(self.movie_features_df)
        super().recommenderData(self.movie_features,self.model_knn,targetTitle)
    
    def load_recommend_LD(self, path = 'models/LM-item-item', targetTitle = 'random', limit =5):
        if(hasattr(self,'movie_features_df_L') == False):
            self.readData()
        self.model_knn = super().openModel(path)
        super().recommend(self.movie_features_df_L,self.model_knn,limit,targetTitle)
    
    def load_recommend_SD(self, path = 'models/SM-item-item', targetTitle = 'random', limit = 5):
        if(hasattr(self,'movie_features_df') == False):
            self.readData()
        
        self.model_knn = super().openModel(path)
        super().recommend(self.movie_features_df,self.model_knn,limit,targetTitle)

    def train_and_save_LD(self, path = 'models/LM-item-item'):
        if(hasattr(self,'movie_features_df_L') == False):
            self.readDataLarge()
        self.createModel(self.movie_features_df_L)
        super().writeModel(path,self.model_knn)

    def train_and_save_SD(self, path = 'models/SM-item-item'):
        if(hasattr(self,'movie_features_df') == False):
            self.readData()   
        self.createModel(self.movie_features_df)
        super().writeModel(path,self.model_knn)

    def load_recommend_SD_TEST(self, path = 'models/SM-item-item', targetTitle = 'random', limit = 5):
        self.model_knn = super().openModel(path)
        super().recommend_test('toystory2RowCollab-small.csv',self.model_knn,limit)

    def __init__(self):
        ModelRecommender.__init__(self)
