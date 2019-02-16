import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class content:

	def __init__(self,data,question,location):
		self.data=data
		self.question=question
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

	def TFIDF_title(self,df,min_df=2):
		df_item=df[['tid','title']]
		df_item['title'] = df_item['title'].fillna("").astype('str')
		df_item.sort_values(by=['tid','title'], ascending=True, inplace=True)
		df_item.drop_duplicates(subset=['tid'],keep='last',inplace=True)
		tf = TfidfVectorizer(stop_words='english',ngram_range=(1,1),min_df=min_df)
		tfidf_matrix = tf.fit_transform(df_item['title'])
		return tfidf_matrix

	def click(self,df,location):
		df_click=df[['uid','tid']]
		df_click['click']=np.ones(df_click.shape[0])
		df_click=df_click.groupby(['uid','tid'],as_index=False).sum()
		click=np.zeros((df_click['uid'].nunique(),df_click['tid'].nunique()))
		for row in df_click.itertuples():
			click[row[1]-1,row[2]-1]=row[3]
		click[location]=-1
		return np.transpose(click)

	def active_time(self,df,location):
		df_time=df[['uid','tid','activeTime']]
		df_time=df_time.fillna(0)
		df_time=df_time.groupby(['uid','tid'],as_index=False).sum()
		active_time=np.zeros((df_time['uid'].nunique(),df_time['tid'].nunique()))
		for row in df_time.itertuples():
			active_time[row[1]-1,row[2]-1]=row[3]
		active_time[location]=-1
		return np.transpose(active_time)

	def nearest(self,rep):
		score = linear_kernel(rep, rep)
		prediction=np.zeros(self.question.shape)
		for uid, tid in zip(self.location[0],self.location[1]):
			yes=score[tid][(self.question[uid]==1).nonzero()].sum()
			no=score[tid][(self.question[uid]==0).nonzero()].sum()
			if yes>no:
				prediction[uid,tid]=1
		return prediction

	def bayes(self):
		pass

	def representation(self, method='click'):
		if method=="category":
			rep=self.TFIDF_category(self.data)
		elif method=="title":
			rep=self.TFIDF_title(self.data)
		elif method=="click":
			rep=self.click(self.data,self.location)
		elif method=="active_time":
			rep=self.active_time(self.data,self.location)
		else:
			rep=0
		return rep

	def predict(self, rep, method='nearest'):
		if method=="nearest":
			return self.nearest(rep)
		elif method=="bayes":
			pass