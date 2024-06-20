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
data = np.loadtxt(open('../../data/jerky_dynamics/Lorenz_exact.txt', "rb"), delimiter=" ", skiprows=0)
t_exact = np.real(data[:,0])
x_exact = np.real(data[:,1])

# # Append xmax at end of x_exact and u_exact
# x_exact = np.append(x_exact, 1.0).reshape(1,-1)
# u_exact = np.append(u_exact,u_exact[0,:].reshape(1,-1),axis=0)

size_xexact = np.shape(x_exact)
size_texact = np.shape(t_exact)

print(size_texact, size_xexact)

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
ffi=0
input_size=1+2*ffi

layer_sizes = [input_size, 16, 16, 16, 1]
step_size = 2e-3
train_iters = 10000
train_iters_lbfgs = 300

iter_arr = np.linspace(0,train_iters,int(train_iters/1000)+1)
obj_arr = np.linspace(0,train_iters,int(train_iters/1000)+1)*0

# Problem parameters
# Domain max and min
tmin0 = 0.0
tmax0 = 50.0

# Number of total points (sampled from these points)

nt = 2001

# number of time steps
nts = 1
nt_local = nts+1

time_adapt = jnp.linspace(tmin0,tmax0,nt_local)
dt = (tmax0-tmin0)/nt

# Arrays for saving iterations and relative errors

tot_train_iters=nts*(train_iters+train_iters_lbfgs)
iter_arr = np.zeros((nts,int((train_iters+train_iters_lbfgs)/10) ))
obj_arr = np.zeros((nts,int((train_iters+train_iters_lbfgs)/10) ))
rel_arr = np.zeros(train_iters+train_iters_lbfgs)

# Divide the time and space domain 

time_adapt = jnp.linspace(tmin0,tmax0,nt_local)

t_step=(tmax0-tmin0)/nts

# parameters
A, B, C, D = 0.4, 2.1, -1.0, 1.0
# initial conditions:
a0, b0, c0 = 0.0, 1.0, 1.0

print(' params = ', A, B, C, D)
print(' init cond = ', a0, b0, c0)

# Here is our initial guess of params:
params = InitializeWeights(layer_sizes, key)

# Arrays to save t and x for postprocessing
t_final = np.array([])

# Set value for continuity requirement (Ncont = M in manuscript)
Ncont = 2
if Ncont ==0:
  def AuxN(t,tmin,tmax):
    tau=(t-tmin)/(tmax-tmin)
    return 1.0-tau

elif Ncont ==1:
  def AuxN(t,tmin,tmax):
    tau=(t-tmin)/(tmax-tmin)
    return 1.0-3.0*tau**2.0 +2*tau**3.0

elif Ncont == 2:
  def AuxN(t,tmin,tmax):
    tau=(t-tmin)/(tmax-tmin)
    return (1.0-6.0*tau**5 + 15*tau**4 - 10.0*tau**3)
    #return (1.0-(np.sin(0.5*jnp.pi*tau))**2/(np.sin(0.5*jnp.pi))**2)
    #return (1.0-(jnp.sin(jnp.pi*tau))**2*(jnp.cos(jnp.pi*tau))**2/(jnp.sin(jnp.pi))**2)

def Fourier_Enc(t,x, N,L):
  arr= jnp.zeros((1+2*N))
  arr = arr.at[0].set(t)
  for i in range(N):
      arr = arr.at[i+1].set(jnp.sin(2.0*x*(i+1)*jnp.pi/L))
      arr = arr.at[i+2].set(jnp.cos(2.0*x*(i+1)*jnp.pi/L))
  return arr

@jit
def f(params, t):
  #inputs = Fourier_Enc(t,x,ffi,L)
  inputs = jnp.array([t])
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
  minind = int((j-1)*size_texact[0]/(nt_local-1))
  maxind = int(j*size_texact[0]/(nt_local-1))

  # Taking mini-batch for optimization and evaluation
  # how many mini-batches equal the entire dataset? N = nx*nt
  batch_size = 256
  batch_size_eval = 256
  
  if(j > 1):
    f_prev = lambda t: f(params_prev, t)

  print('time step = ', j, 'time window = ', tmin, ' to ', tmax)

  # Sub-divide each time window into random temporal collocation points
  #t = random.uniform(key, shape=(nt,), dtype='float64', minval=tmin, maxval=tmax)
  t = jnp.linspace(tmin, tmax, nt)
  
  # prepare the inputs for objective function
  inputs = t[:]

# This is the function we seek to minimize
  @jit
  def objective(params, inputs):
    # These should all be zero at the solution  
    # x_dot = y
    # y_dot = z
    # z_dot = -0.4*z -2.1*y +x*x-1.0
    # x(0) = 0, y(0) = 0, z(0) = 1 

    tin = inputs

    #lambda_t = (10.0*(1.0-(tin-tmin0)/(tmax0-tmin0)) + 1.0).reshape(-1,1)
    lambda_t = (10.0*(1.0-(tin-tmin)/(tmax-tmin)) + 1.0).reshape(-1,1)
    #lambda_t = 1.0

    if (j>1):
      g = lambda t: f(params, t)*(1.0-AuxN(t,tmin,tmax)) + f_prev(t)*AuxN(t,tmin,tmax)
    else:
      g = lambda t: f(params, t)*(1.0-AuxN(t,tmin,tmax)) + (a0 + b0*(t-tmin) + c0*0.5*(t-tmin)**2)*AuxN(t,tmin,tmax)
        
    f_t = jacfwd(g,argnums=0)
    f_tt = jacrev(f_t,argnums=0)
    f_ttt = vmap(jacfwd(f_tt,argnums=0))
    #f_tt = vmap(jacfwd(jacrev(g,argnums=0),argnums=0))
    #f_ttt = vmap(jacrev((jacfwd(jacrev(g,argnums=0),argnums=0)),argnums=0))
    
    fval = vmap(g)(tin).reshape(-1,1)
      
    # Equation loss
    #loss_eq = f_ttt(tin).reshape(-1,1) + A*f_tt(tin).reshape(-1,1) + B*f_t(tin).reshape(-1,1) + C*fval**2 + D 
    loss_eq = f_ttt(tin).reshape(-1,1) + A*vmap(f_tt)(tin).reshape(-1,1) + B*vmap(f_t)(tin).reshape(-1,1) + C*fval**2 + D 
      
    # Return mean loss
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
    solver = LBFGS(fun=objective, value_and_grad=False, has_aux=False, maxiter=500, tol=1e-7, stepsize=0.0)
    res = solver.run(params, inputs=inputs)
    return res.params

  opt_init, opt_update, get_params = adam(step_size, b1=0.9, b2=0.999, eps=1e-8)
  opt_state = opt_init(params)

  #count = 0
  current_min_loss=10e1000  
  for i in range(train_iters):

    # Full batch gradient descent (adam)
    #params, opt_state, value = resnet_update(params, inputs, opt_state)
    
    # Mini batch the inputs data according to batch size
    inputs_batched = inputs[:]
    #inputs_batched = inputs[np.random.choice(inputs.shape[0], size=batch_size)]
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
    inputs_batched = inputs[:]
    #inputs_batched = inputs[np.random.choice(inputs.shape[0], size=batch_size)]
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

    if(rel_loss_ma < 1e-7):
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
    gp = lambda t: f(params, t)*(1.0-AuxN(t,tmin,tmax)) + f_prev(t)*AuxN(t,tmin,tmax)
  else:
    gp = lambda t: f(params, t)*( 1.0 - AuxN(t,tmin,tmax)) + (a0 + b0*(t-tmin) + c0*0.5*(t-tmin)**2)*AuxN(t,tmin,tmax)


  # Computing the time derivatives
  gp_t = vmap(jacfwd(gp,argnums=0))
  gp_tt = vmap(jacfwd(jacrev(gp,argnums=0),argnums=0))

  # Defining ntp and nxp for plotting

  # Using less points so that output files are small in size 
  ntp = size_texact[0]

# Calculate points using space and sub-domain in time window
#  t1 = jnp.linspace(tmin, tmax, ntp)
#  x1 = jnp.linspace(xmin, xmax, nxp)
  tin = t_exact[minind:maxind]
  
  # Exclude the last time data for j>1 
  # Append gp, x and t for plotting

  if(j == 1):
    gp_final = vmap(gp)(tin).reshape(-1)
    gp_final_t = gp_t(tin).reshape(-1)
    gp_final_tt = gp_tt(tin).reshape(-1)
    x_comp = (x_exact[minind:maxind]).reshape(-1)   
    t_final = tin
  else:
    gp_final = np.append(gp_final, vmap(gp)(tin[1:]).reshape(-1))
    gp_final_t = np.append(gp_final_t, gp_t(tin[1:]).reshape(-1))
    gp_final_tt = np.append(gp_final_tt,gp_tt(tin[1:]).reshape(-1))
    x_comp = np.append(x_comp, (x_exact[minind+1:maxind]).reshape(-1))
    t_final = np.append(t_final,tin[1:])

  # Commenting out since we do not need this section any more
  # diff_from_exact = t_step*(vmap(gp)(tin,xin).reshape(-1,1)-f_exact(tin,xin,cvel).reshape(-1,1))
  # exact = t_step*(f_exact(tin,xin,cvel).reshape(-1,1))
  # num += diff_from_exact**2
  # den += exact*exact
  # snum= np.sum(num)
  # sden= np.sum(den)
  #print('MSE = ', np.square(diff_from_exact).mean())
  #print('MSE = ',np.square(diff_from_exact).mean(), file=open('output.txt', 'a'))
    
  print('MSE = ', np.linalg.norm(x_comp - gp_final)/(ntp))

  # Save params for subsequent calculation
  params_prev = params[:][:]
  #tmin_old = tmin

simulation_time = timeit.default_timer() - start
print("The total simulation time is : ", 
  simulation_time, " secs")

# End of loop: Plot the time snapshots

#plt.show()

# Calculate nt final for contour plotting for the whole time domain

ntc = (ntp-1)*nts+1

tc = t_final.reshape(-1)

# Exact and PINN solution on grid (unstructured)

fmgrid = gp_final.reshape(-1)
#fexgrid = f_exact(tc,xc,cvel).reshape(-1)
fexgrid = x_comp.reshape(-1)

# Compute the squared error
error = jnp.sqrt((fmgrid-fexgrid)**2)

# Calculate the L2 error and save it in file
#u_comp = f_exact(tc,xc,cvel).reshape(-1)
L2_error = np.linalg.norm(x_comp - gp_final)/np.linalg.norm(x_comp)
print('L2 error = ', L2_error)

print('\nlayer_sizes = ', layer_sizes, '\nlearning rate = ', step_size, '\nAdam iters = ', train_iters, 
      '\nLBFGS iters = ', train_iters_lbfgs, '\ncollocation points (nx, nt) = ', nt, 
      '\nbatch size = ', batch_size, '\nTime steps = ', nt_local-1, '\nL2 error = ', L2_error , 
      '\nTraining time = ', simulation_time, ' secs',
      file=open('hcspinns_L2_output_jerk_eqn.txt', 'a'))


# 3D phase diagram
ax2 = plt.figure().add_subplot(projection='3d')

xsol = gp_final
ysol = gp_final_t
zsol = gp_final_tt
xyzs = jnp.stack((xsol, ysol, zsol),axis=-1)
#print(xyzs.T)
ax2.plot(*xyzs.T, lw=0.5)
ax2.set_xlabel("X Axis")
ax2.set_ylabel("Y Axis")
ax2.set_zlabel("Z Axis")
#ax2.set_title("Lorenz Attractor")
plt.savefig('hcspinns_jerk_eqn_3D.pdf', bbox_inches='tight')

# Plot x vs t of exact, PINN solution

ax1 = plt.figure().add_subplot()
ax1.plot(t_exact,x_exact,':', linewidth = 2, label='Reference')
ax1.plot(t_final,gp_final,'--', linewidth = 2, label='PINNs')
ax1.legend()
ax1.set_xlabel("Time", fontsize=14)
ax1.set_ylabel("x", fontsize=14)
#ax1.set_title("Lorenz Attractor")
plt.savefig('hcspinn_jerk_eqn_pos.pdf', bbox_inches='tight')

# Save the results into txt file
np.savetxt('hcspinns_jerk_eqn.txt', np.c_[t_final,gp_final])

# Plot loss vs iterations
plt.figure()
for j  in range(1,nt_local):
    plt.semilogy(iter_arr[j-1,:], obj_arr[j-1,:], '-', linewidth = 2, label='iteration', color='blue')
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.savefig('hcspinns_jerk_eqn_convergence.pdf', bbox_inches='tight')
#plt.show()

# Save results in csv format
# Stack the arrays horizontally
stacked_results = np.column_stack((t_final, gp_final, error))
file_path = 'sol.csv'
# Header for each column
header = 't_final, gp_final, error'
np.savetxt(file_path, stacked_results , delimiter=',', header=header)

stacked_residuals = np.column_stack((iter_arr, obj_arr))
file_path = 'loss.csv'
# Header for each column
header = 'iteration, loss'
np.savetxt(file_path, stacked_residuals , delimiter=',', header=header)

print(f"Results saved to csv files")
print("Done!")

