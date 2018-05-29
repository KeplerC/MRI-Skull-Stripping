from dataset_handler import load_data, batch_generator
from autoencoder import CNNAutoencoder
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from config import LEARNING_RATE, NUM_OF_EPOCHS, BATCH_SIZE
import SimpleITK as sitk
def train(X_train, y_train, num_epochs, batch_size, plot=False):
    '''
    Train the model
    @param:
        X_train: the training set
        y_train: the ground truth (skull-stripped version) of the training set
        num_epochs: the number of epoches used for training
        batch_size: the size of a single batch for training
        plot: whether to plot some of the stripped images after training
    '''
    
    print(X_train.shape)
    train_size, img_height, img_width, img_depth = X_train.shape[0],X_train.shape[1],X_train.shape[2],X_train.shape[3]
    
    g = tf.Graph()
    with g.as_default():
          
        # initialize the model
        autoencoder = CNNAutoencoder(img_height,img_width,img_depth,learning_rate=LEARNING_RATE)
        autoencoder.build_graph()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            train_summary_writer = tf.summary.FileWriter("./summary", sess.graph)

            # Initialize all variables
            tf.logging.info('Create new session')
            sess.run(tf.global_variables_initializer())

            # generate batches for training
            batches = batch_generator(X_train, y_train, batch_size, num_epochs, shuffle=True)
           

            print("\n[*] Training ...")
            for batch in batches:
                X_batch, y_batch = zip(*batch)
                
                _train_batch(X_batch,y_batch,sess, autoencoder, train_summary_writer)

            #save the trained model
            saver.save(sess, './my-model')
       
            if plot:
                num_to_plot = 5
                x_sample = X_train[:num_to_plot]
                x_recon = autoencoder.get_reconstructed_img(sess, x_sample)
                
                _plot(sess, x_sample, x_recon,num_to_plot)
    

def _train_batch(X_batch,y_batch,session, model, writer=None, print_every=50):
    '''
    train the model using the given batch
    @param
        X_batch: the batch used for training the model from training set
        y_batch: the ground truth of the batch
        session: the tensorflow session
        model: a CNNAutoencoder instance
        writer: summary writer used for logging
        print_every: the interval to print the value of loss function during training 
    '''
    feed_dict = {
        model.X: X_batch,
        model.y: y_batch
    }

    _, loss, summary = session.run([model.train_op, model.loss, model.loss_summary], feed_dict)

    if writer is not None:
        writer.add_summary(summary, model.step)

    if model.step % print_every == 0 or model.step == 0:
        print("step: %4d\tloss: %.8f" % (model.step, loss))

    model.step += 1


def _plot(sess, figs, recons,num_to_plot):
    '''
    Plot the reconstructed images
    @param:
        sess: current tensorflow session
        figs: the original unstripped images
        recons: the corresponding skull-stripped images learned from figs
        num_to_plot: the number of pairs of images to plot
    '''
    plt.figure(figsize=(20, 12))
    for i in range(num_to_plot):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(figs[i].reshape(256,256),cmap="gray",interpolation="nearest")
        plt.title("Test input")


        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(recons[i].reshape(256,256),cmap="gray",interpolation='nearest')
        plt.title("Reconstruction")
    plt.tight_layout()
    plt.show()



def predict(X_test):
    '''
    Perform skull stripping using trained model on unstripped images.
    Plot the skull-stripped images
    @param:
        X_test: the images to be stripped

    '''
    #restore the trained model
    print(X_test[0].shape)
    tf.reset_default_graph()
    imported_meta = tf.train.import_meta_graph("./my-model.meta")

    with tf.Session() as sess:
        imported_meta.restore(sess,tf.train.latest_checkpoint('./'))
        g = tf.get_default_graph()
        X = g.get_collection("Input")[0]
        recon = g.get_collection("Output")[0]
        feed = {X:X_test}
        bit_masks = sess.run(recon,feed_dict=feed)

        #use bitmask to create predicted image
        predicted = []
        for i in range(len(bit_masks)):
            X_flatten = X_test[i].flatten()
            mask_flatten = bit_masks[i].flatten()
            
            for j in range(len(mask_flatten)):
                
                if mask_flatten[j] <= 0.2:
                    X_flatten[j] = 0
            
            predicted.append(X_flatten.reshape(256,256))

        np.asarray(predicted)

        # plot each original and reconstructed image
        for i in range(len(X_test)):
            plt.subplot(3,1,1)
            plt.imshow(X_test[i].reshape(256,256),cmap="gray")
            plt.imsave(arr=X_test[i].reshape(256,256),fname="/Users/jingyue/Desktop/test_image/origin_{}".format(i),cmap="gray")
            
            plt.subplot(3,1,2)
            plt.imshow(predicted[i].reshape(256,256),cmap="gray")
            plt.imsave(arr=predicted[i].reshape(256,256),fname="/Users/jingyue/Desktop/test_image/stripped_{}".format(i),cmap="gray")

            plt.subplot(3,1,3)
            plt.imshow(bit_masks[i].reshape(256,256),cmap="gray")
            
            plt.show()
            
if __name__ == '__main__':
    
    X_train,y_train,X_test,y_test = load_data(30,"/Users/jingyue/Desktop/MRI_Dataset") 
    # train(X_train,y_train,num_epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE, plot = True)
    predict(X_test)