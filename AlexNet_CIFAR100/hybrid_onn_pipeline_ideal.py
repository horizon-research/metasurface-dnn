import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import cv2 as cv
import datetime
import time
import cupy as cp
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import sparse_categorical_crossentropy
from LoadDataset import get_CIFAR100_color
from CreateModel import create_classifier
from PhaseOptimization import phase_optimization
from keras import __version__ as keras_version
import keras



gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#Global variables that are shared between create_psf_weights() and the backward method of Phase2Kernels
#These control the tiling process (where the PSF is built using the conv weights) and its inverse process
convdim = 3
depth = 48
tile_size = 20
pad = (tile_size-convdim)//2
rows = 4
cols = 12
final_dim = 280
scale = 4
lr_phi = 2.0*(scale**4)
filename_crosstalk = "spectra/identity_matrix.npy"

#Functions that create networks, train them or load their models, and then give the performance metrics of each of the pipeline's steps

def run_create_and_test_models():
    print("Running process to create and train a model:")
    # input image dimensions
    batch_size = 16
    nb_epoch = 200
    learning_rate = 5.0e-3
    lr_decay = 1e-6
    lr_drop = 20

    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR100_color()
    model = create_classifier()
        
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)
    
    sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=False)
    model.compile(optimizer = sgd, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    print("Fitting model...")
    
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0]//batch_size, 
                        epochs=nb_epoch,
                        validation_data=(X_val, y_val),
                        callbacks=[reduce_lr],verbose=1)
    
    print('Saving weights of model...') 
    now = datetime.datetime.now()
    filename_model = "weights/model_color_"+str(now.strftime("%Y-%m-%d-%H-%M"))+".h5"      
    model.save_weights(filename_model)
    print("Done.")
      
    #print('Making predictions for model')          
    #predictions_valid = model.predict(X_val, batch_size=batch_size, verbose=2)
    #score = log_loss(y_val, predictions_valid)
    #print('Score log_loss: ', score)


    #print("Log_loss train: ", score)

    #info_string = 'loss_' + str(score) + '_ep_' + str(nb_epoch)
    #print(info_string)
    
    print('Start Test:')
    test_prediction = model.predict(X_test, batch_size=batch_size, verbose=2)
    score_test = log_loss(y_test, test_prediction)
    print('Test Score log_loss: ', score_test)
    score_test2 = accuracy_score(y_test, np.argmax(test_prediction,axis=1))
    print('Test Score accuracy: ', score_test2)
    
    #weights_step1 = model.get_weights()
    conv_weights = conv_weights = model.layers[0].get_weights()[0]
    print("Saving conv weights into file...")
    now = datetime.datetime.now()
    filename_weights_step1 = "weights/conv_weights_color_"+str(now.strftime("%Y-%m-%d-%H-%M"))+".npy"
    np.save(filename_weights_step1,conv_weights)
    print("Done.")
    return filename_model, filename_weights_step1, score_test, score_test2

def create_psf_weights(filename_weights_step1):
    print("Creating target PSF array...")
    weights = np.load(filename_weights_step1)
    print(weights.shape)
    
    var1 = weights*(weights>=0.0)
    W_conv1_pos = np.abs(var1.squeeze())
    var2 = weights*(weights<0.0)
    W_conv1_neg = np.abs(var2.squeeze())
    kernels_pos_pad = [np.pad(W_conv1_pos[:,:,:,i], ((pad,pad), (pad,pad),(0,0)), 'constant') for i in range(depth)]
    psf_pos = np.concatenate(
        [np.concatenate((kernels_pos_pad[i*cols:(i+1)*cols]), axis=1) for i in range(rows)], axis=0)
    kernels_neg_pad = [np.pad(W_conv1_neg[:,:,:,i], ((pad,pad), (pad,pad),(0,0)), 'constant') for i in range(depth)]
    psf_neg = np.concatenate(
        [np.concatenate((kernels_neg_pad[i*cols:(i+1)*cols]), axis=1) for i in range(rows)], axis=0)

    
    psf_target = np.concatenate((psf_pos, psf_neg), axis=0)
    print(np.shape(psf_target))
    extra_pad_1 = (final_dim - np.shape(psf_target)[0])//2
    extra_pad_2 = (final_dim - np.shape(psf_target)[1])//2
    psf_target = np.pad(psf_target, ((extra_pad_1,extra_pad_1), (extra_pad_2,extra_pad_2),(0,0)), 'constant')
    #dim = np.shape(psf_target)[0]
    norm_factor = np.sum(psf_target) 
    print("Target PSF array constructed!")
    now = datetime.datetime.now()
    filename_psf_target = "PSF_files/target_PSF_"+str(now.strftime("%Y-%m-%d-%H-%M"))+".npy"
    np.save(filename_psf_target, psf_target)
    
    print(np.shape(psf_target))
    print("Finding phase profile whose PSF resembles target...")
    phi,losses = phase_optimization(psf_target,scale=scale,step=lr_phi,iterations=4001)
    phi = np.mod(phi, 2.0*np.pi)
    res = 16
    levels_profile = ((np.rint(phi/(2.0*np.pi/res)))%res)
    phi = levels_profile*(2.0*np.pi/res)
    now = datetime.datetime.now()
    name = "phases/phase_profile_{}x_".format(scale)
    if len(phi.shape)==2:
        filename_phi = name+str(now.strftime("%Y-%m-%d-%H-%M"))+".npy"
    else:
        filename_phi = name+"color_"+str(now.strftime("%Y-%m-%d-%H-%M"))+".npy"
    np.save(filename_phi,phi)
    print("Phase optimization process completed!")
    
    print("Obtaining PSF yielded by phase profile...")
    h = np.fft.ifft2(np.fft.ifftshift(np.exp(1.0j*phi),axes=(0, 1)),axes=(0, 1))
    #h = np.fft.ifft2(np.exp(1.0j*phi),axes=(0, 1))
    psf_new = np.square(np.abs(h))
    if len(psf_new.shape)==2:
        norm = np.sum(psf_new)
    else:
        norm = np.reshape(np.sum(psf_new,axis=(0,1)),(1,1)+psf_new.shape[2:])
    psf_new = psf_new/norm
    psf_new_rescaled = cv.resize(psf_new, (psf_target.shape[0], psf_target.shape[1]), interpolation=cv.INTER_NEAREST)
    now = datetime.datetime.now()
    filename_psf_yielded = "PSF_files/yielded_PSF_"+str(now.strftime("%Y-%m-%d-%H-%M"))+".npy"
    np.save(filename_psf_yielded, psf_new_rescaled)
    
    #Load cross-talk matrix.   
    A = np.load(filename_crosstalk)
    A_t = np.transpose(A)
    P = psf_new_rescaled

    B = np.matmul(A_t.reshape((1,1)+A_t.shape),P.reshape(P.shape+(1,))).reshape(P.shape)
    #The channels of B will be convolved with the channels of the input, and then summed to yield the output signal of the hybrid conv-layer
    
    unpad_1 = extra_pad_1
    unpad_2 = extra_pad_2
    psf_new_unpadded = B[unpad_1:-unpad_1, unpad_2:-unpad_2]
    tiled_kernels = np.split(psf_new_unpadded,2)[0]-np.split(psf_new_unpadded,2)[1]
    print("Extracting weights encoded by yielded PSF, with cross-talk...")
    weights_pm = []
    for i in range(rows):
        for j in range(cols):
            padded_kernel = np.split(np.split(tiled_kernels,rows,axis=0)[i],cols,axis=1)[j]
            kernel = padded_kernel[pad:-pad, pad:-pad]
            weights_pm.append(kernel)
    weights_step2 = np.asarray(weights_pm)
    weights_step2 = weights_step2*norm_factor
    weights_step2 = np.transpose(weights_step2, (1,2,3,0))
    print(weights_step2.shape)
    print("Saving extracted weights...")
    now = datetime.datetime.now()
    filename_weights_step2 = "weights/phasemask_weights_crosstalk_"+str(now.strftime("%Y-%m-%d-%H-%M"))+".npy"
    np.save(filename_weights_step2, weights_step2)
    print("Done.")
    
    return filename_weights_step2, filename_phi, norm_factor, unpad_1, unpad_2

def run_test_loading_models(filename_model, filename_conv_weights=None):
    print("Running process to load and test a model:")
    batch_size = 16
    
    print('Loading weights from {}'.format(filename_model))
    model = create_classifier()
    model.load_weights(filename_model)
    
    if filename_conv_weights:
        conv_bias = model.layers[0].get_weights()[1]
        print('Loading conv weights from {}'.format(filename_conv_weights))
        conv_kernel = np.load(filename_conv_weights)
        model.layers[0].set_weights([conv_kernel,conv_bias])

    print('Start Test')
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR100_color()
    test_prediction = model.predict(X_test, batch_size=batch_size, verbose=2)
    score = log_loss(y_test, test_prediction)
    print('Test Score log_loss: ', score)
    score_test2 = accuracy_score(y_test, np.argmax(test_prediction,axis=1))
    print('Test Score accuracy: ', score_test2)
    
    return score, score_test2

def run_load_finetune_models(filename_model, filename_conv_weights=None):
    print("Running process to load and finetune a model:")
    batch_size = 32
    nb_epoch = 150
    learning_rate = 5.0e-4
    lr_decay = 1e-6
    lr_drop = 20
    if not filename_conv_weights:
        learning_rate = 5.0e-4
    
    print('Loading model from {}'.format(filename_model))
    model = create_classifier()
    model.load_weights(filename_model)
    
    if filename_conv_weights:
        print('Loading conv weights from {}'.format(filename_conv_weights))
        conv_kernel = np.load(filename_conv_weights)
        conv_bias = model.layers[0].get_weights()[1]
        model.layers[0].set_weights([conv_kernel,conv_bias])
        model.layers[0].trainable = False
    
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR100_color()
    #callbacks = [EarlyStopping(monitor='val_loss', patience=15, verbose=0)]
    
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
    
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)
    
    sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=False)
    model.compile(optimizer = sgd, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    print("Fine-tuning model...")
    
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size, 
                        epochs=nb_epoch,
                        validation_data=(X_val, y_val),
                        callbacks=[reduce_lr],
                        verbose=1)
    
    print('Saving weights of model...') 
    now = datetime.datetime.now()       
    filename_model_step3 = "weights/model_color_"+str(now.strftime("%Y-%m-%d-%H-%M"))+".h5"
    model.save_weights(filename_model_step3)
    print("Done.")
      
    print('Making predictions for model')          
    predictions_valid = model.predict(X_val, batch_size=batch_size, verbose=2)
    score = log_loss(y_val, predictions_valid)
    print('Score log_loss: ', score)


    print("Log_loss train: ", score)

    info_string = 'loss_' + str(score) + '_ep_' + str(nb_epoch)
    print(info_string)
    
    print('Start Test:')
    test_prediction = model.predict(X_test, batch_size=batch_size, verbose=2)
    score_test = log_loss(y_test, test_prediction)
    print('Test Score log_loss: ', score_test)
    score_test2 = accuracy_score(y_test, np.argmax(test_prediction,axis=1))
    print('Test Score accuracy: ', score_test2)
    if not filename_conv_weights:
        conv_weights = model.layers[0].get_weights()[0]
        print(conv_weights.shape)
        print("Saving conv weights into file...")
        now = datetime.datetime.now()
        filename_weights_step3 = "weights/conv_weights_color_"+str(now.strftime("%Y-%m-%d-%H-%M"))+".npy"
        np.save(filename_weights_step3,conv_weights)
        print("Done.")
        return filename_model_step3, filename_weights_step3, score_test, score_test2
    return filename_model_step3, score_test, score_test2


#filename_model_step1, filename_weights_step1, log_loss_step1, accuracy_step1 = run_create_and_test_models()
filename_model_step1 = "weights/model_color_2020-09-09-16-51.h5"
filename_weights_step1 = "weights/conv_weights_color_2020-09-09-16-51.npy"
log_loss_step1, accuracy_step1 = run_test_loading_models(filename_model_step1)


filename_weights_step2, filename_phases_step2, norm_factor, unpad_1, unpad_2 = create_psf_weights(filename_weights_step1)
log_loss_step2, accuracy_step2 = run_test_loading_models(filename_model_step1, filename_weights_step2)

filename_model_step3, log_loss_step3, accuracy_step3 = run_load_finetune_models(filename_model_step1, filename_weights_step2)

class Phase2Kernels:
    
    def __init__(self, phi):
        # init the phi to train
        self.phi = phi
    
    def forward(self):
        # input is the tensor phi:(1120,1120,3)
        # clone phi?
        phi = cp.asarray(self.phi)

        h = cp.fft.ifft2(cp.fft.ifftshift(cp.exp(1.0j*phi),axes=(0, 1)),axes=(0, 1))
        psf_new = cp.square(cp.abs(h))
        if len(psf_new.shape)==2:
            norm = cp.sum(psf_new)
        else:
            norm = cp.reshape(cp.sum(psf_new,axis=(0,1)),(1,1)+psf_new.shape[2:])
        psf_new = psf_new/norm

        psf_new_rescaled = cv.resize(cp.asnumpy(psf_new), (final_dim, final_dim), interpolation=cv.INTER_NEAREST)
        psf_new_rescaled = cp.asarray(psf_new_rescaled)
        
        A = np.load(filename_crosstalk)
        A_t = np.transpose(A)
        P = cp.asnumpy(psf_new_rescaled)

        B = np.matmul(A_t.reshape((1,1)+A_t.shape),P.reshape(P.shape+(1,))).reshape(P.shape)
        
        # psf_new_unpadded (152, 228, 3)
        psf_new_unpadded = B[unpad_1:-unpad_1, unpad_2:-unpad_2]
        psf_new_unpadded = cp.asarray(psf_new_unpadded)
        # tiled kernels (76, 228, 3)
        tiled_kernels = cp.split(psf_new_unpadded,2)[0]-cp.split(psf_new_unpadded,2)[1]
        # (48, 3, 3, 3)
        weights_pm = []
        for i in range(rows):
            for j in range(cols):
                padded_kernel = cp.split(cp.split(tiled_kernels,rows,axis=0)[i],cols,axis=1)[j]
                kernel = padded_kernel[pad:-pad, pad:-pad]
                weights_pm.append(kernel)
        #(3,3,3,48)
        
        weights_pm = np.asarray(weights_pm)
        weights_pm = np.transpose(weights_pm, (1,2,3,0))
        
        return weights_pm*norm_factor
    
    def backward(self, grad_out):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input. Thus, we need the gradient of the output with respect to the input,
        i.e. we need the gradient of the kernel tensor with respect to the phase profile.
        """
        # grad_out: gradient of loss function (scalar) wrt kernels, thus, has same shape as kernel tensor
        grad_out = grad_out.numpy()
        #grad_out = cp.asarray(grad_out)
        phi = cp.asarray(self.phi) 
        
        #First, we calculate the gradient of the loss function wrt the PSF
        #This will be in terms of the gradient of the loss function wrt the kernel tensor, grad_out
        def gradient_L_wrt_B(grad_out):
            
            grads_pos_pad = [np.pad(grad_out[:,:,:,i], ((pad,pad), (pad,pad),(0,0)), 'constant') for i in range(depth)]
            grads_pos = np.concatenate(
                [np.concatenate((grads_pos_pad[i*cols:(i+1)*cols]), axis=1) for i in range(rows)], axis=0)
            grads_neg_pad = [np.pad(-1.0*grad_out[:,:,:,i], ((pad,pad), (pad,pad),(0,0)), 'constant') for i in range(depth)]
            grads_neg = np.concatenate(
                [np.concatenate((grads_neg_pad[i*cols:(i+1)*cols]), axis=1) for i in range(rows)], axis=0)
            
            D = np.concatenate((grads_pos, grads_neg), axis=0)
            extra_pad_1 = (final_dim - np.shape(D)[0])//2
            extra_pad_2 = (final_dim - np.shape(D)[1])//2
            
            D = np.pad(D, ((extra_pad_1,extra_pad_1), (extra_pad_2,extra_pad_2),(0,0)), 'constant')
            dim = (final_dim*scale,final_dim*scale)
            D = cv.resize(D, dim, interpolation=cv.INTER_NEAREST)
        
            return D
        
                

        D = gradient_L_wrt_B(grad_out)        
        A = np.load(filename_crosstalk)
        H = np.matmul(A.reshape((1,1)+A.shape),D.reshape(D.shape+(1,))).reshape(D.shape)
        
        H = cp.asarray(H)
        #Now, we can use this gradient to efficiently calculate the gradient of the loss wrt the phases
        h = cp.fft.ifft2(cp.fft.ifftshift(cp.exp(1.0j*phi),axes=(0, 1)),axes=(0, 1))
        gradient_loss = 2.0*cp.imag(cp.exp(-1.0j*phi)*cp.fft.fftshift(cp.fft.fft2((H*h),axes=(0, 1)),axes=(0,1)))
        
        return gradient_loss*norm_factor

def run_end_to_end_models(filename_model, filename_phases):
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR100_color()
    print("Starting end-to-end co-training of phase profile and suffix layers...")

    print("Training dataset shape: ",X_train.shape)
    print("Training label shape:", y_train.shape)
    
    print('Loading model from {}'.format(filename_model))
    model = create_classifier()
    model.load_weights(filename_model)

    EPOCHS = 20
    BS = 64
    INIT_LR = 5e-4
    opt = Adam(lr=INIT_LR)
    PhaseLR = 5e-6
    
    print('Loading phase profile from {}'.format(filename_phases))
    phi_in = np.load(filename_phases)
    phase2Kernel = Phase2Kernels(phi_in)

    def step(X, y):
        # get the weights

        weights = phase2Kernel.forward()
        conv_bias = model.layers[0].get_weights()[1]
        model.layers[0].set_weights([weights,conv_bias])
        
        # keep track of our gradients
        with tf.GradientTape() as tape:
            # make a prediction using the model and then calculate the
            # loss
            pred = model(X)
            print("Pred shape:", pred.shape)
            loss = sparse_categorical_crossentropy(y, pred)
            accuracy = accuracy_score(y, np.argmax(pred,axis=1))
            print("loss",tf.math.reduce_mean(loss))
            print("accuracy:",accuracy)
        grads = tape.gradient(loss, model.trainable_variables)
        print("gradients shapes: ", grads[0].shape, grads[1].shape)
        final_grad = phase2Kernel.backward(grads[0])
        phase2Kernel.phi -= PhaseLR * cp.asnumpy(final_grad)
        
        
        opt.apply_gradients(zip(grads, model.trainable_variables))
        # print(PhaseLR * cp.asnumpy(final_grad))
        # time.sleep(1)

    numUpdates = int(X_train.shape[0] / BS)
# loop over the number of epochs
    for epoch in range(0, EPOCHS):
        # show the current epoch number
        print("[INFO] starting epoch {}/{}...".format(epoch + 1, EPOCHS))
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        sys.stdout.flush()
        epochStart = time.time()
        # loop over the data in batch size increments
        for i in range(0, numUpdates):
            # determine starting and ending slice indexes for the current
            # batch
            print("starting batch: {} of epoch {}".format(i, epoch))
            start = i * BS
            end = start + BS
            # take a step
            step(X_train[start:end], y_train[start:end])
        # show timing information for the epoch
        epochEnd = time.time()
        elapsed = (epochEnd - epochStart) / 60.0
        print("took {:.4} minutes".format(elapsed))
    print("Saving co-trained phases and network parameters...")
    now = datetime.datetime.now()
    filename_model_step4 = "weights/model_co-trained_"+str(now.strftime("%Y-%m-%d-%H-%M"))+".h5"      
    model.save_weights(filename_model_step4)
    filename_phases_step4 = "phases/phases_co-trained"+str(now.strftime("%Y-%m-%d-%H-%M"))+".npy"
    np.save(filename_phases_step4, np.asarray(phase2Kernel.phi))
    print("Done.")
    print("Co-training finished!")
    print('Start Test:')
    test_prediction = model.predict(X_test, batch_size=BS, verbose=2)
    score_test = log_loss(y_test, test_prediction)
    print('Test Score log_loss: ', score_test)
    score_test2 = accuracy_score(y_test, np.argmax(test_prediction,axis=1))
    print('Test Score accuracy: ', score_test2)
    return filename_model_step4, filename_phases_step4, score_test, score_test2
    
filename_model_step4, filename_phases_step4, log_loss_step4, accuracy_step4 = run_end_to_end_models(filename_model_step3, filename_phases_step2)

if __name__ == '__main__':
    print('Keras version: {}\n'.format(keras_version))
    
    print("Metrics of the pipeline's steps:\n\n")
    
    print("Step 1.\nLog loss: {:.4f}, Accuracy:{:.4f} \n".format(log_loss_step1, accuracy_step1))
    print("Step 2.\nLog loss: {:.4f}, Accuracy:{:.4f} \n".format(log_loss_step2, accuracy_step2))
    print("Step 3.\nLog loss: {:.4f}, Accuracy:{:.4f} \n".format(log_loss_step3, accuracy_step3))
    print("Step 4.\nLog loss: {:.4f}, Accuracy:{:.4f} \n".format(log_loss_step4, accuracy_step4))