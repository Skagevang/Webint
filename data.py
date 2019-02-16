import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,recall_score,precision_score

class data:

	def __init__(self, path):
		self.origin_data=self.read_data(path)
		self.remove_frontpage()
		self.gen_click_matrix()
		self.gen_question_key()


	def read_data(self, path):
		"""
		read original data
		stored in self.origin_data
		type: pandas
		"""
		map_lst=[]
		for f in os.listdir(path):
			file_name=os.path.join(path,f)
			if os.path.isfile(file_name):
				for line in open(file_name):
					obj = json.loads(line.strip())
					if not obj is None:
						map_lst.append(obj)
		return pd.DataFrame(map_lst)  

	def remove_frontpage(self):
		"""
		remove frontpanges from self.origin_data.
		"""
		self.origin_data=self.origin_data[self.origin_data["documentId"].notnull()]
		self.status="frontpage removed"

	def numbering(self, df):
		"""
		add two columns: uid and tid
		which transffer userID and documentID into integers starting from 1
		"""
		df = df.sort_values(by=['userId', 'time'])
		new_user = df['userId'].values[1:] != df['userId'].values[:-1]
		new_user = np.r_[True, new_user]
		df['uid'] = np.cumsum(new_user)
		item_ids = df['documentId'].unique().tolist()
		new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(1,len(item_ids)+1)})
		df_merge = pd.merge(df, new_df, on='documentId', how='outer')
		return df_merge

	def gen_click_matrix(self):
		"""
		generate click matrix on self.origin_data
		stored in self.click_matrix
		type: numpy.arrays
		"""
		df=self.origin_data.drop_duplicates(subset=['userId', 'documentId'])
		n_users = df['userId'].nunique()
		n_items = df['documentId'].nunique()
		df_merge=self.numbering(df)
		df_ext = df_merge[['uid', 'tid']]
		self.click_matrix = np.zeros((n_users, n_items))
		for row in df_ext.itertuples():
			self.click_matrix[row[1]-1, row[2]-1] = 1.0

		self.data=pd.merge(self.origin_data,df_merge[['userId','documentId','uid','tid']],on=['userId','documentId'],how='outer').sort_values(by=['userId','time'])

	def gen_question_key(self,fraction=0.2):
		"""
		Erase 20% values from the click matrix, in order to do testing.
		"""
		self.question = self.click_matrix.copy()
		self.key = np.zeros(self.click_matrix.shape)
		location = np.zeros(self.click_matrix.shape)
		for user in range(self.click_matrix.shape[0]):
			size = int(self.click_matrix.shape[1] * fraction)
			test_ratings = np.random.choice(range(self.click_matrix.shape[1]), 
											size=size, 
											replace=False)
			self.question[user, test_ratings] = -1
			self.key[user, test_ratings] = self.click_matrix[user, test_ratings]
			location[user, test_ratings] = 1

		self.location=location.nonzero()

	def evaluate(self,pred):
		pred = pred[self.location].flatten()
		key = self.key[self.location].flatten()
		print("MSE is {:.4f}".format(mean_squared_error(pred, key)))
		print("Precision is {:.4f}".format(precision_score(key, pred)))
		print("Recall is {:.4f}".format(recall_score(key, pred)))

	def show_statistics(self):
		"""
		print statistics of current self.data
		"""
		df=self.data
		num=df.shape[0]
		print("Number of events: {}".format(num))
		num_docs = df['documentId'].nunique()	    
		print("Number of documents: {}".format(num_docs))
		print('Sparsity: {:4.3f}%'.format(float(num) / float(1000*num_docs) * 100))
		df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
		print("Number of events(drop duplicates): {}".format(df.shape[0]))
		print('Sparsity (drop duplicates): {:4.3f}%'.format(float(df.shape[0]) / float(1000*num_docs) * 100))
		
		user_df = df.groupby(['userId']).size().reset_index(name='counts')
		print("Describe by user:")
		print(user_df.describe())
