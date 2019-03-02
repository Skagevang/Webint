import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error,recall_score,precision_score,f1_score,confusion_matrix


class content:

	def __init__(self,data,question,key,location,counter_location):
		self.data=data
		self.question=question
		self.location=location
		self.counter=counter_location
		self.key=key

	def category_list(self,df):
		df_item=df[['tid','category']]
		df_item['category'] = df_item['category'].str.split('|')
		df_item['category'] = df_item['category'].fillna("").astype('str')
		df_item.sort_values(by=['tid','category'], ascending=True, inplace=True)
		df_item.drop_duplicates(subset=['tid'],keep='last',inplace=True)
		df_category=df_item.drop_duplicates(subset=['category'])
		df_category.sort_values(by=['category'],inplace=True)
		df_category['cid']=np.array(range(df_item['category'].nunique()))
		df_merge=pd.merge(df_item, df_category[['category','cid']], on='category', how='outer')
		category_list=np.zeros(df_item.shape[0])
		for row in df_merge[['tid','cid']].itertuples():
			category_list[row[1]-1]=row[2]
		return category_list

	def category_matrix(self,df):
		df_item=df[['tid','category']]
		df_item['category'] = df_item['category'].str.split('|')
		df_item['category'] = df_item['category'].fillna("").astype('str')
		df_item.sort_values(by=['tid','category'], ascending=True, inplace=True)
		df_item.drop_duplicates(subset=['tid'],keep='last',inplace=True)

		df_user=df[['uid','tid','category']]
		df_user.dropna(subset=['category'],inplace=True)
		df_user['category'] = df_user['category'].str.split('|')
		df_user['category'] = df_user['category'].fillna("").astype('str')

		df_category=df_item.drop_duplicates(subset=['category'])
		df_category.sort_values(by=['category'],inplace=True)
		df_category['cid']=np.array(range(df_item['category'].nunique()))

		df_item=pd.merge(df_item, df_category[['category','cid']], on='category', how='outer')
		item_category=np.zeros((df_item.shape[0],df_category.shape[0]))
		for row in df_item[['tid','cid']].itertuples():
			item_category[row[1]-1,row[2]-1]=1

		df_user=pd.merge(df_user, df_category[['category','cid']], on='category', how='inner')
		category_user=np.zeros((df_category.shape[0],df_user['uid'].nunique()))
		deleted=0
		for row in df_user[['cid','uid','tid']].itertuples():
			if self.question[row[2]-1,row[3]-1]!=0:
				category_user[row[1]-1,row[2]-1]+=1
			else:
				deleted+=1
		print(deleted)
		return item_category@category_user

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

	def TFIDF_user_category(self,df):
		df_item=df[['uid','tid','category']]
		df_item['category'] = df_item['category'].str.split('|')
		df_item['category'] = df_item['category'].fillna("").astype('str')
		df_item.sort_values(by=['uid','tid','category'], ascending=True, inplace=True)
		user_category=[]
		last_uid=-1
		last_category=[]
		for row in df_item.itertuples():
			uid=row[1]
			if uid!=last_uid:
				user_category.append(last_category)
				if type(row[3])==list:
					last_category=row[3]
				else:
					last_category=[row[3]]
			else:
				if type(row[3])==list:
					if row[2]==last_tid and last_category[-1]=="":
						last_category=last_category[:-1]
					last_category.extend(row[2])
				else:
					last_category.append(row[2])
			last_uid=uid
			last_tid=row[2]
		user_category=user_category.append(last_category)[1:]
		
		# generate TF-IDF matrix 
		tf = TfidfVectorizer(ngram_range=(1,2),min_df=0)
		tfidf_matrix = tf.fit_transform(user_category)
		return np.transpose(tfidf_matrix)

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
		click.fill(-1)
		for row in df_click.itertuples():
			click[row[1]-1,row[2]-1]=row[3]
		click[location]=0
		return np.transpose(click)

	def active_time(self,df,location):
		df_time=df[['uid','tid','activeTime']]
		df_time=df_time.fillna(0)
		df_time=df_time.groupby(['uid','tid'],as_index=False).sum()
		active_time=np.zeros((df_time['uid'].nunique(),df_time['tid'].nunique()))
		active_time.fill(-1)
		for row in df_time.itertuples():
			active_time[row[1]-1,row[2]-1]=row[3]
		active_time[location]=0
		return np.transpose(active_time)

	def calculate_sim(self,rep,quick=True):
		cosine_sim=np.zeros((rep[0].shape[0],rep[0].shape[0]))
		if quick:
			for r in rep:
				cosine_sim+=cosine_similarity(r)
			return cosine_sim/len(rep)
		for r in rep:
			sim=r@r.transpose()
			r=np.abs(r)
			l=r!=0
			weight=np.zeros(sim.shape)
			for i in range(weight.shape[0]):
				weighted_r=(l*r[i]).transpose()
				weight[i]=r[i]@weighted_r
				del weighted_r
				if i==int(weight.shape[0]/2):
					print("prediction 50% finished")
			cosine_sim+=(sim/weight)/(weight.transpose())
		return cosine_sim/len(rep)


	def nearest(self,cosine_sim,user=False):
		if user:
			question=self.question.transpose()
		else:
			question=self.question
	
		num1=(question==1).sum(1).reshape(-1,1)
		num1[(num1==0).nonzero()]=1
		num2=(question==-1).sum(1).reshape(-1,1)
		num2[(num2==0).nonzero()]=1

		read=(question==1)/num1
		unread=(question==-1)/num2
		num=read+unread

		score=np.dot(question*num,cosine_sim)

		prediction=np.zeros(question.shape)
		prediction.fill(-1)
		prediction[(score>0).nonzero()]=1
		if user:
			return prediction.transpose()
		else:
			return prediction

	def representation(self, method='click'):
		if method=="category":
			rep=self.TFIDF_category(self.data)
		elif method=="user-category":
			rep=self.TFIDF_user_category(self.data)
		elif method=="title":
			rep=self.TFIDF_title(self.data)
		elif method=="click":
			rep=self.click(self.data,self.location)
		elif method=="active_time":
			rep=self.active_time(self.data,self.location)
		else:
			rep=0
		return rep

	def predict(self, rep, method='item', quick=True):
		if type(rep)!=list:
			rep=[rep]
		if quick:
			cosine_sim=self.calculate_sim(rep)
		else:
			cosine_sim=self.calculate_sim(rep,False)

		if method=="item":  # rep = item x embedding
			return self.nearest(cosine_sim)
		elif method=="user": # rep = user x embedding
			return self.nearest(cosine_sim,True)

	def evaluate(self,pred,method):
		# pred: item x user
		if method=="error":
			# pred: prediction
			pred = pred[self.location].flatten()
			key = self.key[self.location].flatten()
			return mean_squared_error(key, pred), precision_score(key, pred), recall_score(key, pred), f1_score(key,pred), confusion_matrix(key,pred)
		elif method=="rank":
			# pred: scores
			key=self.key.copy()
			key[self.counter]=-1
			key=np.transpose(key)
			read=(key==1).sum(1).reshape(key.shape[0],1)
			l=(read==0).nonzero()
			num=(read==0).sum()
			pred=np.transpose(pred)
			pred[self.counter]-=1000
			pred=np.transpose(pred)
			ranking=(-pred).argsort().argsort()
			pred1=np.zeros(pred.shape)
			pred1[ranking<read]=1
			pred.fill(-1)
			pred[ranking<read]=1
			pred=pred.transpose()

			tp=(pred1==key).sum(1)
			read[l]=1
			read=read.reshape(-1)
			recall=(tp/read).sum()/(key.shape[0]-num)
			
			score=((pred1==key)/(ranking+1)).sum(1)
			total_correct=(((key==1)/((np.logical_not(key==1)).argsort().argsort()+1)).sum(1)).reshape(key.shape[0],1)
			total_correct[l]=1
			total_correct=total_correct.reshape(-1)
			arhr=(score/total_correct).sum()/(key.shape[0]-num)

			pred=pred[self.location].flatten()
			key=self.key[self.location].flatten()

			return recall, arhr, mean_squared_error(key, pred), precision_score(key, pred), recall_score(key, pred), f1_score(key,pred), confusion_matrix(key,pred)


		elif method=="user-rank":
			key=self.key.copy()
			key[self.counter]=-1
			read=(key==1).sum(1).reshape(key.shape[0],1)
			l=(read==0).nonzero()
			num=(read==0).sum()
			pred=pred.transpose()
			pred[self.counter]-=1000
			ranking=(-pred).argsort().argsort()
			pred1=np.zeros(pred.shape)
			pred1[ranking<read]=1
			pred.fill(-1)
			pred[ranking<read]=1

			tp=(pred1==key).sum(1)
			read[l]=1
			read=read.reshape(-1)
			recall=(tp/read).sum()/(key.shape[0]-num)

			arhr=((pred1==key)/(ranking+1)).sum()/(key.shape[0]-num)

			pred=pred[self.location].flatten()
			key=self.key[self.location].flatten()
			
			return recall, arhr, mean_squared_error(key, pred), precision_score(key, pred), recall_score(key, pred), f1_score(key,pred), confusion_matrix(key,pred)
