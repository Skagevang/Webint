from data import *

print("started. It may take some time.")
print("===========")

path="active1000"
dataset=data(path)

dataset.remove_frontpage()
print("The number of events after removing frontpage:")
print(dataset.origin_data.shape[0])
print("===========")
### Use dataset.origin_data to get these data.

dataset.remove_nan(["userId","activeTime"])
print("The number of events after removing nan in userId and activeTime:")
print(dataset.data.shape[0])
print("The status of the cleared data:")
print(dataset.status)
print("===========")
### Use dataset.data to get the cleared data.
### Use dataset.status to check the current status of the data.

dataset.gen_click_matrix()
print("The shape of the click matrix:")
print(dataset.click_matrix.shape)
print("The status is still:")
print(dataset.status)
print("===========")
### Use dataset.click_matrix to get the click matrix.

dataset.gen_time_matrix()
print("The shape of the time matrix:")
print(dataset.time_matrix.shape)
print("The status is still:")
print(dataset.status)
print("===========")
### Use dataset.time_matrix to get the time matrix.

train,valid,test=dataset.train_test_split(dataset.click_matrix,valid=True)
print("The shape of the train, valid and test sets are:")
print(train.shape)
print(valid.shape)
print(test.shape)
print("===========")

print("Show some statistics.")
dataset.show_statistics()