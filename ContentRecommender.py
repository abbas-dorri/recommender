from distutils.file_util import move_file
from ModelRecommender import ModelRecommender
import pandas as pd
import re
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class ContentRecommender(ModelRecommender):
    def __init__(self):
        super().__init__()

    def readContentDataGenre_SD(self):
        movies = pd.read_csv('data-small/movies.csv',dtype={'id':'int32','title':'str','genres':'str'})
    
        movies['genres'] = movies['genres'].str.split('|')
        movies = movies.explode('genres')
        
        movies_genres = movies.pivot_table(index='title', columns=['genres'])
        movies_genres[movies_genres.notna()] = 1
        movies_genres = movies_genres.fillna(0)
        return movies_genres

    def readContentDataTag_SD(self):
        movies = pd.read_csv('data-small/movies.csv',dtype={'id':'int32','title':'str','genres':'str'})
        movies_tags = pd.merge(pd.read_csv('data-small/tags.csv',dtype={'id':'int32','title':'str','tag':'str'}), movies, how='inner')
        movies_tags['tag'] = movies_tags['tag'].str.lower()

        movies_tags_grouped = movies_tags.pivot_table(index='title', columns=['tag'])
        movies_tags_grouped[movies_tags_grouped.notna()] = 1
        movies_tags_grouped = movies_tags_grouped.fillna(0)
        return movies_tags_grouped

    def readContentDataGenre_LD(self):
        movies = pd.read_csv('data-big/movies.csv',dtype={'id':'int32','title':'str','genres':'str'})
        
        genreArray = []
        for genres in movies['genres']:
            genres = re.findall("(?<='name': ').+?(?='})", genres)
            genre = ''
            for g in genres:
                genre += g + ' '
            genreArray.append(genre)    
        movies['genres'] = genreArray

        movies = movies.explode('genres')    
        movies_genres = movies.pivot_table(index='title', columns=['genres'])
        movies_genres[movies_genres.notna()] = 1
        movies_genres = movies_genres.fillna(0)
        return movies_genres

    def readContentDataTag_LD(self):
        movies = pd.read_csv('data-big/movies.csv',dtype={'id':'int32','title':'str','genres':'str'})

        movies_tags = pd.merge(pd.read_csv('data-big/keywords.csv',dtype={'id':'int32','title':'str','keywords':'str'}), movies, how='inner')
        keywordArray = []
        for keywords in movies_tags['keywords']:
            keywords = re.findall("(?<='name': ').+?(?='})", keywords)
            keyword = ''
            for g in keywords:
                keyword += g + ' '
            keywordArray.append(keyword)    
        movies_tags['keywords'] = keywordArray
        
        movies_tags['keywords'] = movies_tags['keywords'].str.lower()

        movies_tags_grouped = movies_tags.pivot_table(index='title', columns=['keywords'])
        movies_tags_grouped[movies_tags_grouped.notna()] = 1
        movies_tags_grouped = movies_tags_grouped.fillna(0)
        return movies_tags_grouped

    def readData_LD(self):
        print('reading data')
        movie_features1  = self.readContentDataGenre_LD()
        movie_features2 = self.readContentDataTag_LD()

        self.movie_features_L = movie_features1.merge(movie_features2, on='title', how='inner')
        print(self.movie_features_L)
    
    def readData_SD(self):
        movie_features1  = self.readContentDataGenre_SD()
        movie_features2 = self.readContentDataTag_SD()
        self.movie_features = movie_features1.merge(movie_features2, on='title', how='inner')

    def createModel(self, movie_features : pd.DataFrame):
        print('creating model')

        movie_features_matrix = csr_matrix(movie_features.values)
        model_knn = NearestNeighbors(metric = 'cosine', algorithm='brute')
        model_knn.fit(movie_features_matrix)
        self.model_knn = model_knn
        return model_knn

    def recommend_LD(self, targetTitle = 'random', limit = 5):
        if(hasattr(self,'movie_features_L') == False):
            self.readData_LD()
        self.createModel(self.movie_features_L)
        super().recommend(self.movie_features_L,self.model_knn,limit,targetTitle)
    
    def recommend_SD(self, targetTitle = 'random', limit = 5):
        if(hasattr(self,'movie_features') == False):
            self.readData_SD()
        self.createModel(self.movie_features)
        super().recommend(self.movie_features,self.model_knn,limit,targetTitle)

    def recommend_data_SD(self, targetTitle = 'random'):
        if(hasattr(self,'movie_features') == False):
            self.readData_SD()
        self.createModel(self.movie_features)
        super().recommenderData(self.movie_features,self.model_knn,targetTitle)
        
    def load_recommend_LD(self, path = 'models/LM-content', targetTitle = 'random', limit = 5):
        if(hasattr(self,'movie_features_L') == False):
            self.readData_LD()
        self.model_knn = super().openModel(path)
        super().recommend(self.movie_features_L,self.model_knn,limit,targetTitle)

    def load_recommend_SD(self, path = 'models/SM-content', targetTitle = 'random', limit = 5):
        if(hasattr(self,'movie_features') == False):
            self.readData_SD()

        self.model_knn = super().openModel(path)
        super().recommend(self.movie_features,self.model_knn,limit,targetTitle)

    def train_and_save_LD(self, path = 'models/LM-content'):
        if(hasattr(self,'movie_features_L') == False):
            self.readData_LD()
        self.createModel(self.movie_features_L)
        super().writeModel(path,self.model_knn)

    def train_and_save_SD(self, path = 'models/SM-content'):
        if(hasattr(self,'movie_features') == False):
            self.readData_SD()   
        self.createModel(self.movie_features)
        super().writeModel(path,self.model_knn)

    def load_recommend_SD_TEST(self, path = 'models/SM-content', targetTitle = 'random', limit = 5):
        self.model_knn = super().openModel(path)
        super().recommend_test('toystory2RowContent-small.csv',self.model_knn,limit)