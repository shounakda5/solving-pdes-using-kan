from kan import KAN, LBFGS
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import autograd
from tqdm import tqdm
import time

dim = 2
np_i = 21  # number of interior points (along each dimension)
np_b = 21  # number of boundary points (along each dimension)
ranges = [-1, 1]


def loss_fun(t_int,x_int,t_bc,x_bc,t_ic,x_ic, model):
    input_tensor = torch.cat((t_int, x_int), dim=1)
    u = model(input_tensor)
    u_t = torch.autograd.grad(
            u, t_int,
            grad_outputs=torch.ones_like(t_int),
            create_graph=True)[0]
    u_x = torch.autograd.grad(
            u, x_int,
            grad_outputs=torch.ones_like(x_int),
            create_graph=True)[0]
    u_xx = torch.autograd.grad(
            u_x, x_int,
            grad_outputs=torch.ones_like(x_int),
            create_graph=True)[0]
    R_int = torch.mean(torch.square(u_t + u * u_x - (0.01/torch.pi) * u_xx))
    
    input_tensor = torch.cat((t_bc,x_bc), dim=1)
    u_bc = model(input_tensor)
    R_bc  = torch.mean(torch.square(u_bc))
    
    input_tensor = torch.cat((t_ic,x_ic), dim=1)
    u_ic = model(input_tensor)
    R_ic  = torch.mean(torch.square(u_ic + torch.sin(x_train * torch.pi )))

    return R_int,R_bc,R_ic

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)


# Interior points
sampling_mode = 'mesh'  # 'random' or 'mesh'

N              = 100
t_train        = torch.linspace(0, 0.5, N, requires_grad=True).reshape(N,1)
x_train        = torch.linspace(-1, 1, N, requires_grad=True).reshape(N,1)


if sampling_mode == 'mesh':
    # Generate interior t,x
    t_int, x_int = torch.meshgrid(t_train[1:].squeeze(), x_train[1:-1].squeeze())
    t_int = t_int.reshape(-1, 1)
    x_int = x_int.reshape(-1, 1)
else:
    pass

# Generate boundary t,x
t_bc,x_bc = torch.meshgrid(t_train.squeeze(), torch.cat((x_train[:1], x_train[-1:]), dim=0).squeeze())
t_bc = t_bc.reshape(-1, 1)
x_bc = x_bc.reshape(-1, 1)
# Generate initial t,x
t_ic,x_ic = torch.meshgrid(t_train[:1].squeeze(), x_train.squeeze())
t_ic = t_ic.reshape(-1, 1)
x_ic = x_ic.reshape(-1, 1)

lambda_b       = 10.0
lambda_ic      =10.0

steps = 20
alpha = 0.1
log = 1

loss_int_hist  = np.zeros(steps)
loss_bc_hist    = np.zeros(steps)
loss_ic_hist    = np.zeros(steps)
pred_hist      = np.zeros(N)

model = KAN(width=[2, 3, 3, 2,  1], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25)
optimizer = LBFGS(model.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

def train():
    
    pbar = tqdm(range(steps), desc='description')
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-7)

    for epoch in pbar:
        def closure():
            global loss_int, loss_bc, loss_ic

            # zero the gradient buffers
            optimizer.zero_grad()
            # compute losses
            loss_int,loss_bc,loss_ic = loss_fun(t_int,x_int,t_bc,x_bc,t_ic,x_ic, model)
            loss = loss_int + lambda_b*loss_bc + lambda_ic*loss_ic
            # compute gradients of training loss
            loss.backward()
            
            return loss

        input_tensor = torch.cat((t_int, x_int), dim=1)
        if epoch % 5 == 0 and epoch < 50:
            model.update_grid_from_samples(input_tensor)

        optimizer.step(closure)
        loss = loss_int + lambda_b*loss_bc + lambda_ic*loss_ic

        if epoch % log == 0:
            pbar.set_description("interior pde loss: %.2e | bc loss: %.2e | ic loss: %.2e " % (loss_int.cpu().detach().numpy(), loss_bc.cpu().detach().numpy(), loss_ic.detach().numpy()))

        # print(f'   --- epoch {epoch+1}: loss_int = {loss_int.item():.4e}, loss_bc = {loss_bc.item():.4e}, loss_ic = {loss_ic.item():.4e}')
        
        # save loss
        loss_int_hist[epoch] = loss_int
        loss_bc_hist[epoch] = loss_bc
        loss_ic_hist[epoch] = loss_ic

        
# Measure execution time
start_time = time.time()

train()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

# Visualization (assuming the solution can be visualized similar to the original code)
df = pd.read_excel('Bergurs_Sol.xlsx', sheet_name='Sheet1', header=None)
Num_sol = torch.tensor(df.values, dtype=torch.float32)
t_sample = torch.tensor([0, .1, .2, .3, .4, .5])
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(len(t_sample), 1, figsize=(16, 5 * len(t_sample)))

for l, sample in enumerate(t_sample):
    t_test, x_test = torch.meshgrid(sample, x_train.squeeze())
    t_test = t_test.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    with torch.no_grad():
        pred_hist = model(torch.cat((t_test, x_test), dim=1)).detach().numpy()

    ax[l].plot(x_test.detach().numpy(), pred_hist, marker='o', label='pred. soln')
    ax[l].plot(Num_sol[0,:],Num_sol[l+1,:],'r-',lw=2,label='ref. soln')
    ax[l].set_xlabel('x')
    ax[l].set_ylabel('u')
    ax[l].grid()
    ax[l].legend()
    ax[l].set_title(f'time = {sample:.2f}')

fig.tight_layout()
plt.savefig('Burgers_Equation_KAN.jpeg', format='jpeg')