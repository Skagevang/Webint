from data import *
from content import *

print("Started. It may take some time.")
print("===========")

path="active1000"
dataset=data(path)

print("Shape of the original data:")
print(dataset.origin_data.shape)

print("Shape of the numberred data:")
print(dataset.data.shape)
print("Columns contain:")
print(dataset.data.columns)

print("===========")

print("Overview of the click matrix:")
print(dataset.click_matrix[:5,:5])

print("Overview of the question matrix:")
print(dataset.question[:5,:5])

print("Overview of the key matrix:")
print(dataset.key[:5,:5])

print("See the location of the questions:")
print(dataset.location)

print("Evaluate:")
pred=dataset.click_matrix
dataset.evaluate(pred)

print("===========")

print("Content-based representation:")
content_based=content(dataset.data,dataset.question,dataset.location)
category_rep=content_based.representation('category')
title_rep=content_based.representation('title')
print('The shape of category representation:')
print(category_rep.shape)
print('The shape of title representation:')
print(title_rep.shape)