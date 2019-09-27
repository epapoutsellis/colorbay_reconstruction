#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:20:25 2019

@author: vaggelis
"""

from ccpi.framework import ImageGeometry, AcquisitionGeometry, AcquisitionData

import numpy as np                 
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, FISTA, CGLS

from ccpi.optimisation.operators import BlockOperator, Gradient, SymmetrizedGradient, ZeroOperator, Identity
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction, FunctionOperatorComposition, IndicatorBox

from ccpi.astra.operators import AstraProjectorMC, AstraProjectorSimple, AstraProjector3DSimple, AstraProjector3DMC
from scipy.io import loadmat, savemat
import h5py
from ccpi.plugins.regularisers import FGP_TV, TGV, LLT_ROF

#%% Read Data

pathname = '/media/newhd/shared/DataProcessed/Data_Ryan_Chris/reconstruction/'
filename = 'sinogram_centered.h5'

path = pathname + filename
arrays = {}
f = h5py.File(path)
for k, v in f.items():
    arrays[k] = np.array(v)
XX = arrays['SC']    
X = XX[100:120] # small range of channels
f.close()

# iodine map , relation to number of channel and kev
# y = 0.278 * ch_num + 1.23
# Iodinr K-edge 33.1694
   
#%% Setup Geometries for Colorbay

num_channels = X.shape[0]
num_pixels_h = X.shape[3]
num_pixels_v = X.shape[1]
num_angles = X.shape[2]

# Set angles to use
#angles = numpy.linspace(-numpy.pi,numpy.pi,num_angles,endpoint=False)
angles = np.linspace(-180,180,num_angles,endpoint=False)*np.pi/180

# Define full 3D acquisition geometry and data container.
# NOTE: Default order of labels of the acquisition geometry is 
# [channel, angle, vertical, horizontal]
# Here, we have data with an order of ['channel', vertical, angle, horizontal]
# hence we use the dimension_labels as a last argument

ag = AcquisitionGeometry('cone',
                         '3D',
                         angles,
                         pixel_num_h=num_pixels_h,
                         pixel_size_h=0.25,
                         pixel_num_v=num_pixels_v,
                         pixel_size_v=0.25,                            
                         dist_source_center=233.0, 
                         dist_center_detector=245.0,
                         channels=num_channels, dimension_labels = ['channel', 'vertical', 'angle', 'horizontal'])


# AcquisitionData 3D + channels
data = AcquisitionData(X, ag, dimension_labels = ['channel', 'vertical', 'angle', 'horizontal'] )

# First need the geometric magnification to scale the voxel size relative
# to the detector pixel size.
# magnification factor  =  250/1.76
# dist_source , dist center do not matter

ig = ImageGeometry(voxel_num_x=ag.pixel_num_h, 
                     voxel_num_y=ag.pixel_num_h,
                     voxel_num_z=ag.pixel_num_h,
                     voxel_size_x=0.142, 
                     voxel_size_y=0.142, 
                     voxel_size_z=0.142, 
                     channels=num_channels)


# Setup Astra Projector for 3DMC
A3D_chan = AstraProjector3DMC(ig, ag)


#%% Create callback_method to show date per reconstruction

def show_data_4D(it, obj, x):
    
    plt.imshow(x.as_array()[int(ag.channels/2), int(ag.pixel_num_h/2)], cmap = 'inferno')
    plt.colorbar()
    plt.show()
    
def show_data_3D(it, obj, x):
    
    plt.imshow(x.as_array()[int(ag.pixel_num_h/2)], cmap = 'inferno')
    plt.colorbar()
    plt.show()
    
def show_data_2D(it, obj, x):
    
    plt.imshow(x.as_array(), cmap = 'inferno')
    plt.colorbar()
    plt.show()    
 
#Create show utilities 
    
def show_2D_channel(x, channel_slice):
    
    tmp = x.as_array()
    s1 = len(channel_slice)
    
    fig, axs = plt.subplots(1, s1, constrained_layout=True, figsize = (10,4))
    
    for i in range(s1):
        im = axs[i].imshow(tmp[channel_slice[i]], cmap='inferno')
        axs[i].set_title('Channel {}'.format(channel_slice[i]))
        fig.colorbar(im, ax=axs[i])     
    
def show_4D_channel_slice(x, channel, title, **kwargs):
    
    tmp = x.as_array()
    s = [int(i/2) for i in x.shape]
    axis = kwargs.get('axis', s)  
    
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize = (10,4))

    im1 = axs[0].imshow(tmp[channel,axis[0],:,:], cmap='inferno')
    axs[0].set_title('axial view')
    fig.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(tmp[channel,:,axis[1],:], cmap='inferno')
    axs[1].set_title('coronal view')
    fig.colorbar(im2, ax=axs[1])
    
    im3 = axs[2].imshow(tmp[channel,:,:,axis[2]], cmap='inferno')
    axs[2].set_title('sagittal view')
    fig.colorbar(im3, ax=axs[2])
        
    fig.suptitle(title + ': Channel {}'.format(channel), fontsize = 20)      
    plt.show()   
    
       
    
#%% CGLS reconstruction on the 4D volume = 3D + channels
x_init = A3D_chan.volume_geometry.allocate()

cgls = CGLS(x_init = x_init, operator = A3D_chan, data = data)
cgls.max_iteration = 100
cgls.update_objective_interval = 10
cgls.run(50, verbose = True, callback = show_data_4D)

show_4D_channel_slice(cgls.get_output(), 5, 'CGLS reconstruction') 
show_4D_channel_slice(cgls.get_output(), 10, 'CGLS reconstruction') 
show_4D_channel_slice(cgls.get_output(), 15, 'CGLS reconstruction') 

#%% TV reconstuction channel-wise. 
# Basically for every energy - channel we apply TV reconstruction with the same parameter
# which is not the best 

# Setup Astra Projector for the 3D volume
A3D = A3D_chan.A3D

# Regulariser parameter is the same for all channels
alpha = 0.01

# Reconstruct using FISTA algorithm and gpu option
inner_TV_iter = 50
tolerance = 1e-7
methodTV = 0 # (0 for isotropic TV, 1 for anisotropic TV)
nonnegativity_constraint = 1 # (0 is OFF, 1 is ON)
printing = 0 # (0 is OFF, 1 is ON)
device = 'gpu' # or cpu

g = FGP_TV(alpha, inner_TV_iter, tolerance, methodTV, nonnegativity_constraint, printing, device)

x_init = A3D.volume_geometry.allocate()

# Allocate space for the channel-wise reconstruction
fista_sol_TV_channel_wise = A3D_chan.volume_geometry.allocate()

for i in range(ag.channels):
    
    # Setup L2NormSquarred fidelity term, for each channel
    f = FunctionOperatorComposition(0.5 * L2NormSquared(b = data.subset(channel=i)), A3D)
    
    # Run FISTA 
    fista = FISTA(x_init = x_init, f = f, g = g)
    fista.max_iteration = 100
    fista.update_objective_interval = 50
    fista.run(400, verbose = True, callback = show_data_3D)
    np.copyto(fista_sol_TV_channel_wise.array[i], fista.get_output().array)
    
#%% show reconstruction

show_4D_channel_slice(fista_sol_TV_channel_wise, 5, 'FISTA TV channel-wise reconstruction') 
show_4D_channel_slice(fista_sol_TV_channel_wise, 10, 'FISTA TV channel-wise reconstruction') 
show_4D_channel_slice(fista_sol_TV_channel_wise, 15, 'FISTA TV channel-wise reconstruction') 

#%% Coupling Total variation reconstruction in 4D volume. For this case there is no GPU implementation
# But we can use another algorithm called PDHG ( primal - dual hybrid gradient)

# Set up operators: Projection and Gradient 
op1 = A3D_chan
op2 = Gradient(ig)

# Set up a BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Compute the operator norm
normK = operator.norm()

alpha_coupled = 0.05
f1 = 0.5 * L2NormSquared(b = data)    
f2 = alpha_coupled * MixedL21Norm()

f = BlockFunction(f1, f2) 
g = IndicatorBox(lower = 0)

sigma = 1
tau = 1/(sigma*normK**2)  

pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 100
pdhg.update_objective_interval = 20
pdhg.run(1000, verbose = True, callback = show_data_4D)


#%% Let's move to 2D + energy channel reconstruction

ag2D = AcquisitionGeometry('cone',
                         '2D',
                         angles,
                         pixel_num_h=num_pixels_h,
                         pixel_size_h=0.25,                            
                         dist_source_center=233.0, 
                         dist_center_detector=245.0,
                         channels=num_channels) #, dimension_labels = ['channel', 'horizontal', 'angle'])

ig2D = ImageGeometry(voxel_num_x=ag2D.pixel_num_h, 
                     voxel_num_y=ag2D.pixel_num_h,
                     voxel_size_x=0.142, 
                     voxel_size_y=0.142,  
                     channels=num_channels)


# select a vertical slice
data2DMC = data.subset(vertical=40)
#data2DMC.dimension_labels = ['channel', 'angle', 'horizontal']
# Define astra multichannel projector
A2DMC = AstraProjectorMC(ig2D, ag2D, 'gpu')

# Simple backprojection
backproj = A2DMC.adjoint(data2DMC)
show_2D_channel(backproj, [1, 10])



#%% CGLS 2D + channel reconstruction

x_init = A2DMC.volume_geometry.allocate()

cgls = CGLS(x_init = x_init, operator = A2DMC, data = data2DMC)
cgls.max_iteration = 100
cgls.update_objective_interval = 10
cgls.run(5, verbose = True)

# Show reconstruction 
show_2D_channel(cgls.get_output(), [1,10])


#%% TV 2D + channel reconstruction

# Reconstruct using FISTA algorithm and gpu option
inner_TV_iter = 50
tolerance = 1e-7

# Here isotropic TV : gradient along x, y, and channel directions in isotropic mode
# sqrt( grad_x^2 + grad_y^2 + grad_c^2)
# Anisotropic TV : abs|grad_x| + abs|grad_y| + abs|grad_c|

methodTV = 0 # (0 for isotropic TV, 1 for anisotropic TV)
nonnegativity_constraint = 1 # (0 is OFF, 1 is ON)
printing = 0 # (0 is OFF, 1 is ON)
device = 'gpu' # or cpu

alpha_2DMC = 0.02
g2DMC = FGP_TV(alpha_2DMC, inner_TV_iter, tolerance, methodTV, nonnegativity_constraint, printing, device)

x_init = A2DMC.volume_geometry.allocate()

f2DMC = FunctionOperatorComposition(0.5 * L2NormSquared(b = data2DMC), A2DMC)

# Run FISTA 
fista2DMC = FISTA(x_init = x_init, f = f2DMC, g = g2DMC)
fista2DMC.max_iteration = 100
fista2DMC.update_objective_interval = 50
fista2DMC.run(400, verbose = True)

# Show reconstruction 
show_2D_channel(fista2DMC.get_output(), [1,10])

#%% TGV 2D + channel reconstruction

from ccpi.filters import regularisers

alpha_TGV = 1
beta_TGV = 2

# There is no positivity constraint for the proximal TGV so we override it adding a positivity constrain
def positive_proximal(self, x, tau, out = None):
    
    pars = {'algorithm' : TGV, \
                    'input' : np.asarray(x.as_array(), dtype=np.float32),\
                    'regularisation_parameter':self.regularisation_parameter, \
                    'alpha1':self.alpha1,\
                    'alpha0':self.alpha2,\
                    'number_of_iterations' :self.iter_TGV ,\
                    'LipshitzConstant' :self.LipshitzConstant ,\
                    'tolerance_constant':self.torelance}
            
    res , info = regularisers.TGV(pars['input'], 
          pars['regularisation_parameter'],
          pars['alpha1'],
          pars['alpha0'],
          pars['number_of_iterations'],
          pars['LipshitzConstant'],
          pars['tolerance_constant'],self.device)    
  
    self.info = info
    
    res[res<0] = 0
        
    if out is not None:
        out.fill(res)
    else:
        out = x.copy()
        out.fill(res)
        return out        

setattr(TGV, 'proximal', positive_proximal)
#
g2DMC_TGV = TGV(0.01, alpha_TGV, beta_TGV, 10, 12, 1e-6, 'gpu')
#
## Run FISTA 
fista2DMC_TGV = FISTA(x_init = x_init, f = f2DMC, g = g2DMC_TGV)
fista2DMC_TGV.max_iteration = 100
fista2DMC_TGV.update_objective_interval = 50
fista2DMC_TGV.run(400, verbose = True)
#
show_2D_channel(fista2DMC_TGV.get_output(), [1,10])