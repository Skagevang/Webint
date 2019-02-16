import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class content:

	def __init__(self,data,matrix,location):
		self.data=data
		self.matrix=matrix
		self.location=location

	def TFIDF_category(self,df):
		df_item=df[['tid','category']]
		df_item['category'] = df_item['category'].str.split('|')
		df_item['category'] = df_item['category'].fillna("").astype('str')
		df_item.sort_values(by=['tid','category'], ascending=True, inplace=True)
		df_item.drop_duplicates(subset=['tid'],keep='last',inplace=True)
		
		# generate TF-IDF matrix 
		tf = TfidfVectorizer(ngram_range=(1,2),min_df=0)
		tfidf_matrix = tf.fit_transform(df_item['category'])
		return tfidf_matrix
		
		# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

	def TFIDF_title(self,df,min_df=2):
		df_item=df[['tid','title']]
		df_item['title'] = df_item['title'].fillna("").astype('str')
		df_item.sort_values(by=['tid','title'], ascending=True, inplace=True)
		df_item.drop_duplicates(subset=['tid'],keep='last',inplace=True)
		# generate TF-IDF matrix 
		tf = TfidfVectorizer(stop_words='english',ngram_range=(1,1),min_df=min_df)
		tfidf_matrix = tf.fit_transform(df_item['title'])
		return tfidf_matrix

	def active_time(self):
		pass

	def click(self):
		pass

	def nearest(self):
		pass

	def bayes(self):
		pass

	def representation(self, method='time'):
		if method=="category":
			rep=self.TFIDF_category(self.data)
		elif method=="title":
			rep=self.TFIDF_title(self.data)
		return rep

	def predict(self, method='nearest'):
		pass