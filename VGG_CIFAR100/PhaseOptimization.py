import numpy as np
import cupy as cp
import cv2 as cv
mempool = cp.get_default_memory_pool()

def phase_optimization(psf_target,scale=1,step=1.0,iterations=100,amplitude=1.0):
    print("Performing phase optimization...")
    #A = amplitude
    if scale==1:
        Y = psf_target
        Y= cp.asarray(Y)
    else:
        dim_0 = scale*psf_target.shape[0]
        dim_1 = scale*psf_target.shape[1]
        dim = (dim_0,dim_1)
        Y= cv.resize(psf_target, dim, interpolation=cv.INTER_NEAREST)
        Y= cp.asarray(Y)
    if len(Y.shape)==2:
        Y = Y/cp.sum(Y)
    else:
        Y = Y/cp.reshape(cp.sum(Y,axis=(0,1)),(1,1)+Y.shape[2:])
    
    phi = cp.random.random_sample(Y.shape)*2.0*np.pi
    #phi = cp.fft.fftshift(cp.angle(cp.fft.fft2(cp.sqrt(Y),axes=(0, 1))),axes=(0,1))+2.0*np.pi
    losses = []
    #grads = []
    mempool.free_all_blocks()
    for i in range(iterations):
        
        h = cp.fft.ifft2(cp.fft.ifftshift(cp.exp(1.0j*phi),axes=(0, 1)),axes=(0, 1))
        psf_new = cp.square(cp.abs(h))
        if len(psf_new.shape)==2:
            norm = cp.sum(psf_new)
        else:
            norm = cp.reshape(cp.sum(psf_new,axis=(0,1)),(1,1)+psf_new.shape[2:])
        psf_new = psf_new/norm
        h = h/cp.sqrt(norm)
        
        D = (Y-psf_new)
        loss = cp.sum(cp.square(D))
        losses.append(loss)
        if i%100==0:
            print("Iteration {} of {}, Loss: {}".format(i,iterations,loss))
        
        gradient = -4.0*cp.imag(cp.exp(-1.0j*phi)*cp.fft.fftshift(cp.fft.fft2((D*h),axes=(0, 1)),axes=(0,1)))
        phi = phi - step*gradient
        #grads.append(cp.sum(gradient**2))
        mempool.free_all_blocks()
        
    print("Phase Optimization process completed!")
    return cp.asnumpy(phi), np.array(losses)

