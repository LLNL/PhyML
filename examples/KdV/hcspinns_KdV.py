import jax.numpy as jnp
import numpy as np
from jax import random
from jax import jit, vmap
import jax.nn as jnn
from jax.example_libraries.optimizers import adam
#from functools import partial
from jax import value_and_grad
from jax import jacfwd, jacrev
from jaxopt import LBFGS
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from scipy import io
from jax import config
import timeit
config.update("jax_enable_x64", True)

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


# reference data
data = io.loadmat('../../data/KdV/KdV.mat')  
u_exact = np.real(data['uu'])
t_exact = np.real(data['tt'])    
x_exact = np.real(data['x'])

data = np.loadtxt(open('../../data/KdV/kdv_exact_1s.txt', "rb"), delimiter=" ", skiprows=0)
u_FD_1s = np.real(data[:,1])
x_FD = np.real(data[:,0])

# Append xmax at end of x_exact and u_exact
x_exact = np.append(x_exact, 1.0).reshape(1,-1)
u_exact = np.append(u_exact,u_exact[0,:].reshape(1,-1),axis=0)

size_xexact = np.shape(x_exact)
size_texact = np.shape(t_exact)

start = timeit.default_timer()

key = random.PRNGKey(42)

# Use Xavier/Glorot intializer

initializer = jnn.initializers.glorot_uniform()

# Initialization of params with random numbers

def InitializeWeights(layer_sizes, seed):
    weights = []
    for m, n, in zip(layer_sizes[:-1], layer_sizes[1:]):
        w = initializer(seed, (m, n), dtype=jnp.float64)
        b = jnp.zeros((n,))        
        weights.append([w,b])

    return weights

# Hyperparameters.

# Fourier feature integer
ffi=1
input_size=1+2*ffi

layer_sizes = [input_size, 32, 32, 32, 32, 1]
step_size = 2e-3
train_iters = 20000
train_iters_lbfgs = 200
step = step_size
iter_arr = np.zeros(int(train_iters/1000)+1)

iter_arr = np.linspace(0,train_iters,int(train_iters/1000)+1)
obj_arr = np.linspace(0,train_iters,int(train_iters/1000)+1)*0
rel_arr = np.zeros(train_iters+train_iters_lbfgs)

# Problem parameters
# Domain max and min
tmin0 = 0.0
tmax0 = 1.0

xmin = -1.0
xmax = 1.0

# Number of total points (sampled from these points)

nt = 1001
nx = 1001

# number of time steps
nts = 10
nt_local = nts+1

# Arrays for saving iterations and relative errors

tot_train_iters=nts*(train_iters+train_iters_lbfgs)
iter_arr = np.zeros((nts,int((train_iters+train_iters_lbfgs)/10) ))
obj_arr = np.zeros((nts,int((train_iters+train_iters_lbfgs)/10) ))
rel_arr = np.zeros(train_iters+train_iters_lbfgs)

# Divide the time and space domain 

time_adapt = jnp.linspace(tmin0,tmax0,nt_local)
x = random.uniform(key, shape=(nx,), dtype='float64', minval=xmin, maxval=xmax)

t_step=(tmax0-tmin0)/nts

#f_prev = lambda x,t: jnp.sin(x)
tmin_old = -1 # some arbitrary negative number

gamma1 = 1.0
gamma2 = 0.0025

# Here is our initial guess of params:
params = InitializeWeights(layer_sizes, key)

# Arrays to save t and x for postprocessing
t_final = np.array([])
x_final = np.array([])

# Set value for continuity requirement (Ncont = M in manuscript)
Ncont=0
if Ncont ==0:
  def AuxN(t,tmin,tmax):
    tau=(t-tmin)/(tmax-tmin)
    return 1.0-tau

elif Ncont ==1:
  def AuxN(t,tmin,tmax):
    tau=(t-tmin)/(tmax-tmin)
    return 1.0-3.0*tau**2.0 +2*tau**3.0

def Fourier_Enc(t,x, N,L):
  arr= jnp.zeros((1+2*N))
  arr = arr.at[0].set(t)
  for i in range(N):
      arr = arr.at[i+1].set(jnp.sin(2.0*x*(i+1)*jnp.pi/L))
      arr = arr.at[i+2].set(jnp.cos(2.0*x*(i+1)*jnp.pi/L))
  return arr

# Define L for periodic function
L = (xmax-xmin)

@jit
def f(params, t, x):
  # xmin = 0.0
  # xmax = jnp.pi
  inputs = Fourier_Enc(t,x,ffi,L)
  for w, b in params:
    outputs = jnp.dot(inputs, w) + b  # Linear transform
    inputs = jnp.tanh(outputs) 
  return outputs

diff_from_exact=0.0

# Counter for iteration and loss array
counter = 0

for j  in range(1,nt_local):

  # Counter for iteration and loss array
  counterl = 0

  tmin = time_adapt[j-1]
  tmax = time_adapt[j]

  # For comparing with exact solution
  minind = int((j-1)*size_texact[1]/(nt_local-1))
  maxind = int(j*size_texact[1]/(nt_local-1))

  # Taking mini-batch for optimization and evaluation
  # how many mini-batches equal the entire dataset? N = nx*nt
  batch_size = 256
  batch_size_eval = 256
  
  if(j > 1):
    f_prev = lambda t, x: f(params_prev, t, x)

  print('time step = ', j, 'time window = ', tmin, ' to ', tmax)

  # Sub-divide each time window into random temporal collocation points
  t = random.uniform(key, shape=(nt,), dtype='float64', minval=tmin, maxval=tmax)
  
  
  # prepare the inputs for objective function
  tp, xp = jnp.meshgrid(t, x)
  inputs = jnp.stack([tp.flatten(), xp.flatten()], 1)

# This is the function we seek to minimize
  @jit
  def objective(params, inputs):
    tin = inputs[:,0]
    xin = inputs[:,1]

    lambda_t = (10.0*(1.0-(tin-tmin0)/(tmax0-tmin0)) + 1.0).reshape(-1,1)
    #lambda_t = (10.0*(1.0-(tin-tmin)/(tmax-tmin)) + 1.0).reshape(-1,1)
    #lambda_t = 1.0

    if (j>1):
      g = lambda t,x: f(params, t, x)*(1.0-AuxN(t,tmin,tmax)) + f_prev(t, x)*AuxN(t,tmin,tmax)
    else:
      g = lambda t,x: f(params, t, x)*( 1.0 - AuxN(t,tmin,tmax)) + (jnp.cos(jnp.pi*x))*AuxN(t,tmin,tmax)
        
    f_t = vmap(jacfwd(g,argnums=0))
    f_x = vmap(jacfwd(g,argnums=1))
  
    f_xxx = vmap(jacrev((jacfwd(jacrev(g,argnums=1),argnums=1)),argnums=1))

    fval = vmap(g)(tin,xin).reshape(-1,1)   
    
    loss_eq = f_t(tin,xin).reshape(-1,1) + gamma1*fval*f_x(tin,xin).reshape(-1,1) + gamma2*f_xxx(tin,xin).reshape(-1,1) 
    
    return jnp.mean(lambda_t*loss_eq**2) 
    

  # Adam optimizer 
  @jit
  def resnet_update(params, inputs, opt_state):
      """ Compute the gradient for a batch and update the parameters """
      value, grads = value_and_grad(objective)(params, inputs)
      opt_state = opt_update(0, grads, opt_state)
      return get_params(opt_state), opt_state, value
  
  @jit
  def resnet_update_bfgs(params, inputs):
    """ Compute the gradient for a batch and update the parameters using bfgs """
    solver = LBFGS(fun=objective, value_and_grad=False, has_aux=False, maxiter=500, tol=1e-6, stepsize=0.0)
    res = solver.run(params, inputs=inputs)
    #params, opt_state = res
    return res.params

  opt_init, opt_update, get_params = adam(step_size, b1=0.9, b2=0.999, eps=1e-8)
  opt_state = opt_init(params)

  #count = 0
  current_min_loss=10e1000  
  for i in range(train_iters):

    # Full batch gradient descent (adam)
    #params, opt_state, value = resnet_update(params, inputs, opt_state)
    
    # Mini batch the inputs data according to batch size
    inputs_batched = inputs[np.random.choice(inputs.shape[0], size=batch_size)]
    params, opt_state, value = resnet_update(params, inputs_batched, opt_state)
   
    # Evaluate in a larger batch size 
    inputs_eval_batched = inputs[np.random.choice(inputs.shape[0], size=batch_size_eval)]
    loss = objective(params, inputs_eval_batched)

    # Save the params for lowest loss
    if(loss < current_min_loss):
      current_min_loss=loss
      save_params=params[:][:]

    if loss < 1e-12: 
      print("Iterations converged at iteration = ", i, 'objective function = ', loss)
      break

    if i % 1000 == 0:
          print("Iteration {0:3d} objective {1}".format(i,loss))

    # Save loss and iteration counter in an array     
    if i % 10 == 0:
      obj_arr[j-1,counterl] = loss
      iter_arr[j-1,counterl]  = counter*10
      counter = counter+1 
      counterl = counterl+1 

  # Use the save_params (lowest loss) for subsequent lbfgs iterations
      
  params=save_params[:][:]

  #Perform lbfgs after adam
  print('Perform L-BFGS iterations for batch size = ', batch_size)
  prev_loss = 1e100
  rel_loss = 0.0

  for i in range(train_iters_lbfgs):
    
    # Full batch gradient descent (lbfgs) Caution: very expensive!
    #params, opt_state, value = resnet_update_bfgs(params, inputs, opt_state)
    
    # Mini batch the inputs data according to batch size
    inputs_batched = inputs[np.random.choice(inputs.shape[0], size=batch_size)]
    params = resnet_update_bfgs(params, inputs_batched)
    
    # Evaluate on different collocation points with a larger batch size
    inputs_eval_batched = inputs[np.random.choice(inputs.shape[0], size=batch_size_eval)]
    loss_error = objective(params, inputs_eval_batched)
    
    if(loss_error < current_min_loss):
      current_min_loss=loss_error
      save_params=params[:][:] 

    # Compute relative error and moving average (last 5 iterations)
    relative_loss = np.abs(loss_error-prev_loss)
    rel_arr[i] = relative_loss
    rel_loss_ma = np.average(rel_arr[i-4:i+1]) 

    if(rel_loss_ma < 1e-6):
      print("Iterations converged at iteration = ", i, 'objective function = ', loss, 'relative loss = ', rel_loss_ma)
      break
    
    # Save loss for computing relative error
    prev_loss = loss_error

    if i % 50 == 0:
      print("Iteration {0:3d} objective {1} relloss {2}".format(i, loss_error, rel_loss_ma))

    # Save loss and iteration counter in an array    
    if i % 10 == 0:
      obj_arr[j-1,counterl] = loss_error
      iter_arr[j-1,counterl]  = counter*10
      counter = counter+1 
      counterl = counterl+1 
  
  #Use best params so far
  params=save_params[:][:]
      
  # Postprocess using the best params

  if (j>1):
    gp = lambda t,x: f(params, t, x)*(1.0-AuxN(t,tmin,tmax)) + f_prev(t,x)*AuxN(t,tmin,tmax)
  else:
    gp = lambda t,x: f(params, t, x)*( 1.0 - AuxN(t,tmin,tmax)) + (jnp.cos(jnp.pi*x))*AuxN(t,tmin,tmax)

  # Defining ntp and nxp for plotting

  # Using less points so that output files are small in size 
  #ntp = int((nt-1)/nts)+1
#  ntp = 101
#  nxp = 101
  ntp = size_texact[1]
  nxp = size_xexact[1]


  # Calculate points using space and sub-domain in time window
#  t1 = jnp.linspace(tmin, tmax, ntp)
#  x1 = jnp.linspace(xmin, xmax, nxp)
  t1 = t_exact[0][minind:maxind]
  x1 = jnp.linspace(xmin, xmax, nxp)
  

  # Exclude the last time data for j>1 
  if(j == 1):
    tp, xp = jnp.meshgrid(t1, x1)
  else:
    tp, xp = jnp.meshgrid(t1[1:], x1)

  # Inputs for plotting
  inputsp = jnp.stack([tp.flatten(), xp.flatten()], 1)

  tin = inputsp[:,0]
  xin = inputsp[:,1]

  # Append gp, x and t for plotting

  if(j == 1):
    gp_final = vmap(gp)(tin, xin).reshape(-1)
    u_comp = (u_exact[:,minind:maxind]).reshape(-1)   
    x_final = xin
    t_final = tin
  else:
    gp_final = np.append(gp_final, (vmap)(gp)(tin, xin).reshape(-1))
    u_comp = np.append(u_comp, (u_exact[:,minind+1:maxind]).reshape(-1))
    x_final = np.append(x_final,xin)
    t_final = np.append(t_final,tin)

  # Commenting out since we do not need this section any more
  # diff_from_exact = t_step*(vmap(gp)(tin,xin).reshape(-1,1)-f_exact(tin,xin,cvel).reshape(-1,1))
  # exact = t_step*(f_exact(tin,xin,cvel).reshape(-1,1))
  # num += diff_from_exact**2
  # den += exact*exact
  # snum= np.sum(num)
  # sden= np.sum(den)
  #print('MSE = ', np.square(diff_from_exact).mean())
  #print('MSE = ',np.square(diff_from_exact).mean(), file=open('output.txt', 'a'))
    
  print('MSE = ', np.linalg.norm(u_comp - gp_final)/(ntp*nxp))

  tminp = jnp.stack([jnp.ones(nxp)*tmin, x1], 1)
  tmaxp = jnp.stack([jnp.ones(nxp)*tmax, x1], 1)

  # Plot exact and PINN solution for 4 time snapshots
  if(j == 1):
    plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(x1, vmap(gp)(tminp[:,0], tminp[:,1]), '-',label='PINN, t=0')
    plt.plot(x1, vmap(gp)(tmaxp[:,0], tmaxp[:,1]), ':',label='PINN, t=0.25')
    plt.plot(x_exact.T[:], u_exact[:,int(tmin*200)], '--',label='Reference, t=0')
    plt.plot(x_exact.T[:], u_exact[:,int(tmax*200)], '-.',label='Reference, t=0.25')
    plt.legend()
    plt.xlim([xmin, xmax])

  elif(j == (nt_local-1)/2):
    ax2 = plt.subplot(2, 2, 2)
    plt.plot(x1, vmap(gp)(tmaxp[:,0], tmaxp[:,1]), ':',label='PINN, t=0.5')
    plt.plot(x_exact.T[:], u_exact[:,int(tmax*200)], '-.',label='Reference, t=0.5')    
    plt.legend()
    plt.xlim([xmin, xmax])

  elif(j == (nt_local-1)*3/4):
    ax3 = plt.subplot(2, 2, 3)
    plt.plot(x1, vmap(gp)(tmaxp[:,0], tmaxp[:,1]), ':',label='PINN, t=0.75')
    plt.plot(x_exact.T[:], u_exact[:,int(tmax*200)], '-.',label='Reference, t=0.75')    
    plt.legend()
    plt.xlim([xmin, xmax])
  elif(j == nt_local-1):
    ax3 = plt.subplot(2, 2, 4)
    plt.plot(x1, vmap(gp)(tmaxp[:,0], tmaxp[:,1]), ':',label='PINN, t=1.0')
    plt.plot(x_exact.T[:], u_exact[:,int(tmax*200)], '-.',label='Reference, t=1.0')    
    plt.legend(loc='lower center')
    plt.xlim([xmin, xmax])
    
  # Save params for subsequent calculation
  params_prev = params[:][:]
  tmin_old = tmin

simulation_time = timeit.default_timer() - start
print("The total simulation time is : ", 
  simulation_time, " secs")

# End of loop: Plot the time snapshots

plt.savefig('hcspinns_KdV_eqn_timestep.pdf')
#plt.show()

# Calculate nt final for contour plotting for the whole time domain

ntc = (ntp-1)*nts+1

# t1 = jnp.linspace(tmin0, tmax0, ntc)
# x1 = jnp.linspace(xmin, xmax, nxp)

tc = t_final.reshape(-1)
xc = x_final.reshape(-1)

# Exact and PINN solution on grid (unstructured)

fmgrid = gp_final.reshape(-1)
#fexgrid = f_exact(tc,xc,cvel).reshape(-1)
fexgrid = u_comp.reshape(-1)

# Compute the squared error
error = jnp.sqrt((fmgrid-fexgrid)**2)

# Calculate the L2 error and save it in file
#u_comp = f_exact(tc,xc,cvel).reshape(-1)
L2_error = np.linalg.norm(u_comp - gp_final)/np.linalg.norm(u_comp)
print('L2 error = ', L2_error)

print('\nlayer_sizes = ', layer_sizes, '\nlearning rate = ', step_size, '\nAdam iters = ', train_iters, 
      '\nLBFGS iters = ', train_iters_lbfgs, '\ncollocation points (nx, nt) = ', nx, nt, 
      '\nbatch size = ', batch_size, '\nTime steps = ', nt_local-1, '\nL2 error = ', L2_error , 
      '\nTraining time = ', simulation_time, ' secs',
      file=open('hcspinns_L2_output_KdV.txt', 'a'))

# Plot contours of exact, PINN solution and error

plt.figure(figsize=(8,12))
ax = plt.subplot(3,1,1)

im = plt.tripcolor(tc, xc, fexgrid, shading='gouraud', cmap='rainbow')
#ax.set_aspect(1.0)
ax.set(xlim=(tmin0, tmax0), ylim=(xmin, xmax))
plt.ylabel("Exact", fontsize=14)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax = plt.subplot(3,1,2)

im = plt.tripcolor(tc, xc, fmgrid, shading='gouraud', cmap='rainbow')
#ax.set_aspect(1.0)
ax.set(xlim=(tmin0, tmax0), ylim=(xmin, xmax))
plt.ylabel("PINN", fontsize=14)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=12)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax = plt.subplot(3,1,3)
im = plt.tripcolor(tc, xc, error, shading='gouraud', cmap='rainbow')

#ax.set_aspect(1.0)
ax.set(xlim=(tmin0, tmax0), ylim=(xmin, xmax))
plt.ylabel("Error", fontsize=14)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=12)
plt.savefig('hcspinns_KdV_contour.pdf', bbox_inches='tight')

# Plot loss vs iterations
plt.figure()
for j  in range(1,nt_local):
    plt.semilogy(iter_arr[j-1,:], obj_arr[j-1,:], '-', linewidth = 2, label='iteration', color='blue')
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.savefig('hcspinns_KdV_convergence.pdf', bbox_inches='tight')
#plt.show()

# Save results in csv format
# Stack the arrays horizontally
stacked_results = np.column_stack((t_final, x_final, gp_final, error))
file_path = 'sol.csv'
# Header for each column
header = 't_final,x_final,gp_final,error'
np.savetxt(file_path, stacked_results , delimiter=',', header=header)

stacked_residuals = np.column_stack((iter_arr, obj_arr))
file_path = 'loss.csv'
# Header for each column
header = 'iteration, loss'
np.savetxt(file_path, stacked_residuals , delimiter=',', header=header)

print(f"Results saved to csv files")
print("Done!")

