import matlab.engine as engine
import numpy as np

M = np.load('../../examples/data/M_opt.npy')
q = np.load('../../examples/data/q_opt.npy')
z = np.load('../../examples/data/z_opt.npy').T
w = np.load('../../examples/data/w_opt.npy').T
u = np.load('../../examples/data/u_opt.npy').T
x = np.load('../../examples/data/x_opt.npy').T
N = M.shape[0]

eng = engine.start_matlab()

# error in optimal control
ctr_linear_error = []  # w - M*z - q
ctr_compl_error = []  # w*z

# error in simulaton
sim_linear_error = []  # w - M*z - q
sim_compl_error = []  # w*z

for i in range(N):
    Mi = M[i, ...]
    qi = q[i, :].reshape(-1, 1)
    wi = w[i, :].reshape(-1, 1)
    zi = z[i, :].reshape(-1, 1)
    ui = u[i, :].reshape(-1, 1)
    xi = x[i, :].reshape(-1, 1)
    
    ctr_linear_error.append((wi - Mi @ zi - qi).squeeze().tolist())
    ctr_compl_error.append((wi * zi).squeeze().tolist())
    
    ans = eng.LCP(Mi, qi)
    z_solved = np.array(ans).squeeze()
    w_solved = np.array(Mi @ z_solved + qi.squeeze())
    sim_linear_error.append((w_solved - Mi @ z_solved - qi.squeeze().tolist()))
    sim_compl_error.append((w_solved * z_solved).tolist())
    
    import pdb; pdb.set_trace()

import pdb; pdb.set_trace()
pass
