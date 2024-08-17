import pandas as pd
import copy
import numpy as np
from scipy.spatial.distance import cdist



df = pd.read_csv("./Data/Iris.csv")

class centroid_ration:

    def __init__(self, df, ground_true, predicted, matrix):
        # self.matrix = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=int)
        # self.matrix = copy.deepcopy(matrix)
        self.df = copy.deepcopy(df)
        self.ground_true = copy.deepcopy(ground_true)
        self.predicted = copy.deepcopy(predicted)
        self.pairs = []
        self.ready = False

    # def find_pairs(self):
    #     def remove_row_col(arr, i, j):
    #         arr = np.delete(arr, i, axis=0)
    #         arr = np.delete(arr, j, axis=1)
    #         return arr
    #     def old_row_calc(deleted_row, new_row):
    #         ls = deleted_row[::-1]
    #         for i in ls:
    #             if i <= new_row:
    #                 new_row += 1
    #         return new_row
    #     def old_col_calc(deleted_col, new_col):
    #         ls = deleted_col[::-1]
    #         for i in ls:
    #             if i <= new_col:
    #                 new_col += 1
    #         return new_col

    #     ground_true_list = []
    #     predicted_list = []
    #     true_unique = self.ground_true.unique()
    #     predicted_unique = np.unique(self.predicted)
    #     for i in true_unique:
    #         ground_true_list.append(self.df.iloc[self.ground_true[self.ground_true == i].index, :].mean())

    #     for i in predicted_unique:
    #         predicted_list.append(self.df.iloc[self.predicted == i, :].mean()) 

    #     ground_true_array = np.array(ground_true_list)
    #     predicted_array = np.array(predicted_list)

    #     distance_matrix = cdist(ground_true_array, predicted_array, metric='euclidean') 
        
    #     temp_matrix = copy.deepcopy(distance_matrix)
    #     deleted_col = []
    #     deleted_row = []
    #     score = 0
    #     while(not 0 in temp_matrix.shape):
    #         flattened_index = np.argmin(temp_matrix)
    #         row, col = np.unravel_index(flattened_index, temp_matrix.shape)
    #         old_row = old_row_calc(deleted_row, row)
    #         old_col = old_col_calc(deleted_col, col)
    #         deleted_col.append(col)
    #         deleted_row.append(row)
    #         # score += matrix[old_row, old_col]#####################################################
    #         temp_matrix = remove_row_col(temp_matrix, row, col)
    #     return score



    # def score(self):
    #     if self.ready:
    #         return np.sum(self.matrix[[pair[0] for pair in self.pairs], [pair[1] for pair in self.pairs]])
    #     return -1

    def score(self):
        def remove_row_col(arr, i, j):
            arr = np.delete(arr, i, axis=0)
            arr = np.delete(arr, j, axis=1)
            return arr
        def old_row_calc(deleted_row, new_row):
            ls = deleted_row[::-1]
            for i in ls:
                if i <= new_row:
                    new_row += 1
            return new_row
        def old_col_calc(deleted_col, new_col):
            ls = deleted_col[::-1]
            for i in ls:
                if i <= new_col:
                    new_col += 1
            return new_col
        def create_contingency_matrix(ground_true_unique, predicted_unique):
            contingency_matrix = np.zeros((len(ground_true_unique), len(predicted_unique)), dtype=int)
            for i, label1 in enumerate(ground_true_unique):
              for j, label2 in enumerate(predicted_unique):
                contingency_matrix[i, j] = np.sum((self.ground_true == label1) & (self.predicted == label2))
            return contingency_matrix

        ground_true_list = []
        predicted_list = []
        true_unique = np.unique(self.ground_true)
        predicted_unique = np.unique(self.predicted)
        contingency_matrix = create_contingency_matrix(true_unique, predicted_unique)

        for i in true_unique:
            ground_true_list.append(self.df.iloc[self.ground_true[self.ground_true == i].index, :].mean())
        for i in predicted_unique:
            predicted_list.append(self.df.iloc[self.predicted == i, :].mean()) 
        ground_true_array = np.array(ground_true_list)
        predicted_array = np.array(predicted_list)
        distance_matrix = cdist(ground_true_array, predicted_array, metric='euclidean')   
        temp_matrix = copy.deepcopy(distance_matrix)
        deleted_col = []
        deleted_row = []
        score = 0
        while(not 0 in temp_matrix.shape):
            flattened_index = np.argmin(temp_matrix)
            row, col = np.unravel_index(flattened_index, temp_matrix.shape)
            old_row = old_row_calc(deleted_row, row)
            old_col = old_col_calc(deleted_col, col)
            deleted_col.append(col)
            deleted_row.append(row)
            score += contingency_matrix[old_row, old_col]
            temp_matrix = remove_row_col(temp_matrix, row, col)
        return score
    
ground = df.iloc[:,5]

predicted = np.zeros((len(ground)), dtype=int)
for i in range(len(ground)):
    random = np.random.randint(0, len(ground.unique()))
    predicted[i] = random

df = df.iloc[:, 1:5]

# ground = predicted
ground = pd.Series(ground)

model = centroid_ration(df, ground, predicted, None)
model.score()