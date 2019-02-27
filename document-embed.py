import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable, backward
import numpy as np



class batch_data:

	def __init__(self,batch_size,category_list,click_rep,time_rep,question):
		self.batch_size=batch_size
		self.click=click_rep
		self.time=time_rep
		self.category=category_list
		self.question=question


	def read_batch(self,num):
		total_size=self.click.shape[0]
		if num*self.batch_size>total_size:
			return [],[],[],[],[],1
		click=self.click[self.batch_size*num:self.batch_size*(num+1),:]
		time=self.time[self.batch_size*num:self.batch_size*(num+1),:]
		category=self.category[self.batch_size*num:self.batch_size*(num+1)]
		target=self.question[self.batch_size*num:self.batch_size*(num+1)]
		location=np.zeros(target.shape)
		location[target.nonzero()]=1
		# weight the unread items lower
		location[(target==-1).nonzero()]=0.8
		return torch.from_numpy(click).float(), torch.from_numpy(time).float(), torch.from_numpy(category).long(), torch.from_numpy(target).float(), torch.from_numpy(location).float(), 0



class module(nn.Module):
	
	def __init__(self,user_num,category_num,dimension):
		super(module, self).__init__()
		self.embedding=nn.Embedding(category_num,dimension)
		self.linear_c=nn.Linear(user_num,dimension,False)
		self.linear_t=nn.Linear(user_num,dimension,False)
		self.gru=nn.GRU(dimension*3,dimension,num_layers=2,bias=False)
		# self.softlinear=nn.Linear(dimension,user_num,False)
		
	def forward(self,click,time,category,target,location):
		click=self.linear_c(click)
		time=self.linear_t(time)
		category=self.embedding(category)
		gru_input=torch.cat((click,time,category),1).unsqueeze(0)
		# gru_input=torch.cat((click,time),1).unsqueeze(0)
		hidden,_=self.gru(gru_input)
		hidden=hidden.squeeze(0)

		pred=nn.Tanh()(hidden)
		# pred= self.softlinear(pred)

		lost=(pred-target)**2
		num=location.sum(1).unsqueeze(1)
		lost=lost*location
		lost=lost/num
		err=lost.sum()
		err=err/lost.shape[0]

		# Criterion=nn.MSELoss(size_average=True)
		# err=Criterion(pred, target)

		return hidden.data.numpy(),err,pred.data.numpy()


class embedding:

	def __init__(self,dimension,batch_size,max_iter,content,lr):
		self.dimension=dimension
		self.max_iter=max_iter

		self.content=content
		
		self.category=content.category_list(content.data)
		self.click=content.representation('click')
		self.time=content.representation('active_time')

		self.question=np.transpose(content.question)
		self.key=np.transpose(content.key)
		
		self.data=batch_data(batch_size,self.category,self.click,self.time,self.question)
		self.save_folder='embedding'
		self.log='embedding/log'

		self.module=module(self.click.shape[1],int(self.category.max())+1,self.dimension)
		self.module=self.module
		self.module_optimizer=optim.SGD(self.module.parameters(),lr,momentum=0.9)
		
		if self.log!="":
			with open(self.log,"w") as log:
				log.write("")

	def save(self):
		torch.save(self.module.state_dict(),self.save_folder+"/model"+str(self.iter)+".pkl")
		# print("finished saving")

	def test(self):
		self.module.eval()
		location=np.zeros(self.key.shape)
		location[self.key.nonzero()]=1
		hidden,err,pred=self.module(torch.from_numpy(self.click).float(),torch.from_numpy(self.time).float(),torch.from_numpy(self.category).long(),torch.from_numpy(self.key).float(),torch.from_numpy(location).float())
		
		if self.iter>=10:
			torch.save(hidden,self.save_folder+"/hidden"+str(self.iter)+".pkl")
		
		err=err.data.item()
		print("Testing err "+str(err))
		if self.log!="":
			with open(self.log,"a") as log:
				log.write("Testing err "+str(err)+"\n")

		r,arhr=self.content.evaluate(pred,method="rank")

		pred=self.content.predict(hidden)
		MSE, precision, recall, f1, confusion_matrix=self.content.evaluate(pred,method="error")
		print("\nEvaluation Scores:")
		print("Hit: {:.4f}, Hit+Rank: {:.4f}".format(r,arhr))
		print("MSE: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}\n".format(MSE,precision,recall,f1))
		print("Prediction: Negative  Positive")
		print("||Not Read:{}".format(confusion_matrix[0]))
		print("||||||Read:{}\n".format(confusion_matrix[1]))
		if self.log!="":
			with open(self.log,"a") as log:
				log.write("\nEvaluation Scores:\n")
				log.write("Hit: {:.4f}, Hit+Rank: {:.4f}\n".format(r,arhr))
				log.write("MSE: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}\n".format(MSE,precision,recall,f1))
				log.write("Prediction: Negative  Positive\n")
				log.write("||Not Read:{}\n".format(confusion_matrix[0]))
				log.write("||||||Read:{}\n".format(confusion_matrix[1]))


	def train(self):
		self.iter=0
		print("iter  "+str(self.iter))
		self.test()
		while True:
			sum_err=0
			self.iter+=1
			print("\niter  "+str(self.iter))
			with open(self.log,"a") as log:
				log.write("\niter  "+str(self.iter)+"\n")
			num=0
			End=0
			while End==0:
				click, time, category, target, location, End = self.data.read_batch(num)
				
				if End==1:
					break
				click=Variable(click)
				time=Variable(time)
				category=Variable(category)
				target=Variable(target)
				location=Variable(location)

				self.module_optimizer.zero_grad()

				self.module.train()
				hidden,err,pred=self.module(click,time,category,target,location)
				err.backward()
				sum_err+=err.data.item()
				
				self.module_optimizer.step()
				num+=1

			if num in [30,50,80]:
				print(str(int(num*self.batch_size/20344))+"%")
			print("Trainig err "+str(sum_err/(num)))
			# self.save()
			if self.log!="":
				with open(self.log,"a") as log:
					log.write("Training err "+str(sum_err/(num))+"\n")
			self.test()
			if self.iter==self.max_iter:
				break



if __name__ == '__main__':
	from data import *
	from content import *
	dimension=1000
	batch_size=128
	max_iter=20
	lr=0.01

	path="active1000"
	dataset=data(path)
	c=content(dataset.data,dataset.question,dataset.click_matrix,dataset.location,dataset.counter_location)
	model=embedding(dimension,batch_size,max_iter,c,lr)
	print("Start Training")
	model.train()
    