import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class data:

	def __init__(self, path):
		self.origin_data=self.read_data(path)
		self.status="original"

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

	def remove_nan(self,category="userId"):
		"""
		'nan' values removed according to given category
		parameters: "all" or category's name or list
		stored in self.data
		type: pandas
		"""
		if category=="all":
			self.data=self.origin_data.dropna(how="any")
			self.status="all none removed"
		elif type(category)==str:
			self.data = self.origin_data[self.origin_data[category].notna()]
			self.status=category+" removed"
		else:
			self.data=self.origin_data.dropna(how="any",subset=category)
			self.status=", ".join(category)+" removed"

	def show_statistics(self):
	    """
	    print statistics of current self.data
	    """
	    df=self.data.sort_values(by=['userId', 'time'], ascending=True)
	    num=df.shape[0]
	    print("Number of remaining events: {}".format(num))
	    num_docs = df['documentId'].nunique()	    
	    print("Number of remaining documents: {}".format(num_docs))
	    print('Sparsity: {:4.3f}%'.format(float(num) / float(1000*num_docs) * 100))
	    df.drop_duplicates(subset=['userId', 'documentId'], inplace=True)
	    print("Number of remaining events(drop duplicates): {}".format(df.shape[0]))
	    print('Sparsity (drop duplicates): {:4.3f}%'.format(float(df.shape[0]) / float(1000*num_docs) * 100))
	    
	    user_df = df.groupby(['userId']).size().reset_index(name='counts')
	    print("Describe by user:")
	    print(user_df.describe())

	def numbering(self, df):
		"""
		add two columns: uid and tid
		which transffer userID and documentID into integers starting from 1
		"""
		df = df.sort_values(by=['userId', 'documentId'])
		new_user = df['userId'].values[1:] != df['userId'].values[:-1]
		new_user = np.r_[True, new_user]
		df['uid'] = np.cumsum(new_user)
		item_ids = df['documentId'].unique().tolist()
		new_df = pd.DataFrame({'documentId':item_ids, 'tid':range(1,len(item_ids)+1)})
		df_merge = pd.merge(df, new_df, on='documentId', how='outer')
		return df_merge


	def gen_click_matrix(self):
		"""
		generate click matrix based on self.data
		stored in self.click_matrix
		type: numpy.arrays
		"""
		df=self.data.drop_duplicates(subset=['userId', 'documentId'])
		n_users = df['userId'].nunique()
		n_items = df['documentId'].nunique()
		df_merge=self.numbering(df)
		df_ext = df_merge[['uid', 'tid']]
		self.click_matrix = np.zeros((n_users, n_items))
		for row in df_ext.itertuples():
			self.click_matrix[row[1]-1, row[2]-1] = 1.0

	def gen_time_matrix(self):
		"""
		generate activetime matrix based on self.data
		stored in self.time_matrix
		type: numpy.arrays
		"""
		if 'activeTime' not in self.status:
			raise("remove nan values in activeTime first!")
		df=self.data[['userId','documentId','activeTime']].groupby(['userId','documentId'],as_index=False).sum()
		n_users = df['userId'].nunique()
		n_items = df['documentId'].nunique()
		df_merge=self.numbering(df)
		df_ext = df_merge[['uid', 'tid', 'activeTime']]
		self.time_matrix = np.zeros((n_users, n_items))
		for row in df_ext.itertuples():
			self.time_matrix[row[1]-1, row[2]-1] = row[3] 


	def train_test_split(self,matrix,fraction=0.2,valid=False):
		"""
		train test split, based on given matrix
		return 2 or 3 matrix
		"""
		m=matrix.copy()
		np.random.shuffle(m)
		l=m.shape[0]
		if valid:
			train=m[:int(l*(1-2*fraction))]
			valid=m[int(l*(1-2*fraction)):int(l*(1-fraction))]
			test=m[int(l*(1-fraction)):]
			return train, valid, test
		else:
			train=m[:int(l*(1-fraction))]
			test=m[int(l*(1-fraction)):]
			return train, test
