#split dataset to training and test sets.
import numpy as np
import tensorflow  as tf 
import matplotlib as plt
import pydicom 
import os
from random import shuffle


def load_data(num_of_patients,path):
    '''
    Load the dataset and generate normalized training and testing sets.
    We treat the MRI images of each patient as an unit, which means all images of a patient will either be in training or in test set.
    @param
        num_of_patients: the number of patients used in this model
        path: the dataset's path
    
    @return:
        X_train: the training set
        y_train: the ground truth (skull-stripped version) of the training set
        X_test: the testing set
        y_test: the ground truth (skull-stripped version) of the test set
    '''
    patient_ids = [i for i in range(num_of_patients)]
    shuffle(patient_ids)
    print(patient_ids)
    train_idx = patient_ids[:num_of_patients-1]
    test_idx = patient_ids[-1]

    # gather training data
    X_train = []
    y_train = []
    for idx in train_idx:
        X_temp = []
        y_temp = []
        X_train_dir = os.path.join(path,str(idx),"origin/")
        for f in os.listdir(X_train_dir):
            if f.endswith(".dcm"):
                X_temp.append(f)
        X_temp.sort()
        for file_name in X_temp:
            file_array = pydicom.read_file(X_train_dir + file_name).pixel_array
            img_height = file_array.shape[0]
            img_width = file_array.shape[1]
            square_file_array = np.zeros((img_height,img_height))
            diff = (img_height - img_width)//2
            if diff != 13:
                print(X_train_dir + file_name)
            square_file_array[:,diff:diff+img_width] = file_array

            #add pixels at the beginning and end of the image
            # file_array = np.append(file_array,np.zeros((img_height,diff)),axis = 1)
            # test = np.zeros((img_height,diff))
            # print(file_array.shape)
            # file_array = np.insert(file_array,1,np.zeros((img_height,diff)),axis=1)
            X_train.append(square_file_array)
       
        y_train_dir = os.path.join(path,str(idx),"stripped/")
        for f in os.listdir(y_train_dir):
            if f.endswith(".dcm"):
                y_temp.append(f)
        y_temp.sort()

        for file_name in y_temp:
            file_array = pydicom.read_file(y_train_dir + file_name).pixel_array
            img_height = file_array.shape[0]
            img_width = file_array.shape[1]
            square_file_array = np.zeros((img_height,img_height))
            diff = (img_height - img_width)//2
            if diff != 13:
                print(y_train_dir + file_name)

            square_file_array[:,diff:diff+img_width] = file_array

            #add pixels at the beginning and end of the image
            # file_array = np.append(file_array,np.zeros((img_height,diff)),axis = 1)
            # test = np.zeros((img_height,diff))
            # print(file_array.shape)
            # file_array = np.insert(file_array,1,np.zeros((img_height,diff)),axis=1)
            y_train.append(square_file_array)


    # gather test data
    X_test = []
    y_test = []
    X_test_dir = os.path.join(path,str(test_idx),"origin/")
    y_test_dir = os.path.join(path,str(test_idx),"stripped/")
    
    X_test_files = []
    y_test_files = []
    for f in os.listdir(X_test_dir):
        if f.endswith(".dcm"):
            X_test_files.append(f)
    X_test_files.sort()

    for file_name in X_test_files:
        file_array = pydicom.read_file(X_test_dir + file_name).pixel_array
        img_height = file_array.shape[0]
        img_width = file_array.shape[1]
        square_file_array = np.zeros((img_height,img_height))
        diff = (img_height - img_width)//2
        if diff != 13:
            print(X_test_dir + file_name)

        square_file_array[:,diff:diff+img_width] = file_array
        X_test.append(square_file_array)
            
    for f in os.listdir(y_test_dir):
        if f.endswith(".dcm"):
            y_test_files.append(f)            
    y_test_files.sort()

    for file_name in y_test_files:
        file_array = pydicom.read_file(y_test_dir + file_name).pixel_array
        img_height = file_array.shape[0]
        img_width = file_array.shape[1]
        square_file_array = np.zeros((img_height,img_height))
        diff = (img_height - img_width)//2
        if diff != 13:
            print(y_test_dir + file_name)
        square_file_array[:,diff:diff+img_width] = file_array
        y_test.append(square_file_array)

    X_train = np.expand_dims(np.asarray(X_train),axis = 4)
    y_train = np.expand_dims(np.asarray(y_train),axis = 4)
    X_test = np.expand_dims(np.asarray(X_test),axis= 4)
    y_test = np.expand_dims(np.asarray(y_test),axis = 4)
    
    #perform normalization
    X_train = X_train.astype('float32') /255.
   
    y_train = y_train.astype('float32') /255.
    X_test = X_test.astype('float32') /255.
    y_test = y_test.astype('float32') /255.
    
    return X_train,y_train,X_test,y_test



def batch_generator(X, y, batch_size, num_epochs, shuffle=True):
    '''
    Generate batches from training set to train the model
    @param:
        X: the entire training set
        y: the ground truth of training set
        batch_size: the size of a single batch used for training
        num_epochs: the number of epoches used for training
        shuffle: whether to shuffle the training set when generating batches

    @return:
        return one batch at a time
    '''
    data_size = X.shape[0]
    num_batches_per_epoch = data_size // batch_size + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            X_shuffled = X[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            X_shuffled = X
            y_shuffled = y

        for batch_num in range(num_batches_per_epoch - 1):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            X_batch = X_shuffled[start_index:end_index]
            y_batch = y_shuffled[start_index:end_index]
            batch = list(zip(X_batch, y_batch))

            yield batch