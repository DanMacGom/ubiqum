import os
import numpy as np
import pandas as pd
from pathlib import Path

path_handler = {}

for word in (str(Path.cwd()) + "\\3.- IoT Analytics\\Wifi Locationing\\Datasets").split("\\"):
    if word in path_handler:
        path_handler[word] += 1
    else:
        path_handler[word] = 1

print(path_handler)
datasets_path = "\\".join(path_handler.keys())

train = pd.read_csv(datasets_path + "\\trainingData.csv")
validation = pd.read_csv(datasets_path + "\\validationData.csv")

print(train)


# TODO: complete this class
class PaginationHelper:

    # The constructor takes in an array of items and a integer indicating
    # how many items fit within a single page
    def __init__(self, collection, items_per_page):
        self.collection = collection
        self.items_per_page = items_per_page

    # returns the number of items within the entire collection
    def item_count(self):
        return len(self.collection)

    # returns the number of pages
    def page_count(self):
        return len(self.collection) // self.items_per_page + (len(self.collection) % self.items_per_page)

    # returns the number of items on the current page. page_index is zero based
    # this method should return -1 for page_index values that are out of range
    def page_item_count(self, page_index):
        pass

    # determines what page an item is on. Zero based indexes.
    # this method should return -1 for item_index values that are out of range
    def page_index(self, item_index):
        pass


helper = PaginationHelper(range(1, 25), 10)
helper.page_count()
helper.item_count()