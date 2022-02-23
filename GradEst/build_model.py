from __future__ import absolute_import, division, print_function 
import tensorflow as tf
import numpy as np
from numpy.core._multiarray_umath import ndarray
from tensorflow import keras




class ImageModel():

    def __init__(self, model_name, dataset_name, train = False, load = False, **kwargs):

        if dataset_name=='gtsrb':
            print("USING GTSRB MODEL")
            self.model_name=keras.models.load_model("/home/junfeng/FSE/Hard_Label/saved_models/GTSRB_model.h5")
            self.num_classes =43
            self.r=0.09

        if dataset_name=='mnist':
            print("USING MNIST MODEL")
            self.model_name=keras.models.load_model("/home/junfeng/FSE/Hard_Label/saved_models/mnist.h5")
            self.num_classes =10
            self.r=0.2

        elif dataset_name=='cifar10':
            self.model_name = keras.models.load_model("/home/junfeng/FSE/Hard_Label/saved_models/resnet_56.h5")
            self.r = 0.02

        self.dataset_name = dataset_name
        self.data_model = dataset_name + model_name
        self.framework = 'keras'
        self.input_size = 32

        self.channel = 3

        self.model_p = keras.models.load_model("/home/junfeng/FSE/Hard_Label/double_NN/saved_models/model_r")

        self.t = 0.11


    def predict(self, x, d="m", batch_size = 500, logits = False):

        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)

        if len(x.shape) == 4:
            pass

        if d=="f":


            return np.argmax(self.model_name.predict(x),1)



        if d=="t":

            self.num=50


            num=x.shape[0]
            self.noise = np.random.uniform(-self.r, self.r, size=(self.num,
                                                                x.shape[1],
                                                                x.shape[2],
                                                                x.shape[3]))

            x_rep = np.repeat(x, repeats=self.num, axis=0)

            for i in range(num):
                x_rep[i * self.num:(i + 1) * self.num] = x_rep[i * self.num:(i + 1) * self.num] \
                                                         + self.noise


            x_rep = np.clip(x_rep, 0, 1)

            pre = np.argmax(self.model_name.predict(x_rep), axis=1)

            y = np.empty(num)

            for i in range(num):
                y[i] = np.bincount(pre[i * self.num:(i + 1) * self.num]).argmax()

            return y


        if d=="m":



            prediction = self.model_name.predict(x)
            #
            pre_sort = np.sort(prediction, axis=1)[:, -2:]

            # pre_sort_sum=np.sum(pre_sort,axis=1)
            #

            ticket = self.model_p.predict(pre_sort)[:, 1]

            label = np.argmax(prediction, 1)

            pred = np.empty_like(label)
            original_label = np.eye((prediction.shape[-1]))[label]
            second = np.argmax(prediction * (1 - original_label), 1)


            coin= np.sum(x, axis=(1, 2, 3))*10000
            # d = np.random.uniform(0, 1, x.shape[0])


            #coin = np.random.uniform(0, 1, x.shape[0])


            for i in range(prediction.shape[0]):

                # gap=np.abs(prediction[i][label[i]]-second_c[i][0])
                b = ticket[i]
                # if ticket[i]>=0.4:

                # if True:
                if b >= self.t:

                    if int(coin[i])%2==0 :
                        # if int(np.sum(x[i])*100)%2==0:
                        # if:q
                        # .random.uniform(0,1,1)<=d[i]:

                        pred[i] = second[i]


                    else:

                        pred[i] = label[i]
                    # print(f"Give True Prediction: {label[i]} \n")



                else:
                    pred[i] = label[i]
            return pred

    def predict_t(self, x,):

        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)

        if len(x.shape) == 4:
            pass

        return self.model_name.predict(x)







