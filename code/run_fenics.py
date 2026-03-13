from dolfin import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

set_log_active(False)

cmap2 = mpl.cm.get_cmap("spring")
cmap = mpl.cm.get_cmap("winter")

# Define the problem
Length = 40 # 40.0
nx = 200 # 200
nT = 900 # 900
keep = 5
mesh = IntervalMesh(nx, 1, Length)
P1 = FiniteElement('P', interval, 1)
element = MixedElement([P1, P1])
V = FunctionSpace(mesh, element)

density_tip = np.zeros((int(nT / keep), nx))
density_in = np.zeros((int(nT / keep), nx))

# Define the parameters
T = 40 # 90 (hours)
dt = T/nT
alphval = 0.039 # 0.039
cval = 0.6
# Define the initial condition
u0 = Expression(('0.8*exp(-0.1*pow(abs(x[0]-7),2))',0), degree=1)
#define radial coordinate
rexp = Expression(('1/x[0]',0),degree=1)
r = interpolate(rexp, V)
invr, _ = split(r)
# Define other elements

u_n = interpolate(u0, V)
u = Function (V)
n, rho = split(u)
n_n, rho_n= split(u_n)
v_1, v_2 = TestFunctions(V)
k = Constant((1,))

s_scal = Constant(0.2) # 0.2
beta = Constant(0.022) # Constant(0.022)
alpha = Constant(alphval)
D = Constant(0.02) # Constant(0.02)

n_mid = 0.5*(n_n + n)
v_sval = 0.235 # 0.235
v_s = Constant((v_sval,))


F = n*v_1*dx - n_n*v_1*dx 
#Diffusion coefficient
F += D*dot(grad(n), grad(v_1))*dt*dx
# To account for cylindrical coordinates
F += invr*D*dot(grad(n), Constant((1,)))*v_1*dt*dx
F += dt*dot(grad(n_mid),v_s)*v_1*dx
# To account for cylindrical coordinates
F += invr*dt*n_mid*Constant(v_sval)*v_1*dx



F -=dt*s_scal*n*v_2*dx-(rho-rho_n)*v_2*dx
F += beta*v_1*rho*(n)*dt*dx
F -= alpha*(n)*v_1*dt*dx

# Apply the boundary conditions
u_L = (Constant(0.0),Constant(0.0))
u_R = (Constant(0.0),Constant(0.0))
bc_L = DirichletBC(V, u_L, 'near(x[0], 1.0)')
bc_R = DirichletBC(V, u_R, f'near(x[0], {Length})')
bcs = [bc_L, bc_R]

# bc = PeriodicBC(V, Boundary())# Solve the problem
t = 0.0
a, L = lhs(F), rhs(F)
V2 = FunctionSpace(mesh, 'P', 1)
X = np.linspace(1,Length,nx)
poss = []

fix,axs = plt.subplots(2,1)
i=0
for t in np.linspace(0,T,nT):

    solve(F==0, u,bcs)
    u_n.assign(u)
    _n, _rho= u.split()
    n_field = interpolate(_n, V2)

    if i%keep==0:# i%100==0:
        rho_field = interpolate(_rho, V2)
        if i%keep==0:
            axs[0].plot(X,[n_field(x) for x in X],color=cmap2(t/T))
            axs[1].plot(X,[rho_field(x) for x in X],color=cmap(t/T))

        density_tip[int(i / 10), :] = [n_field(x) for x in X]
        density_in[int(i / 10), :] = [rho_field(x) for x in X]
        
    poss.append(np.argmax([n_field(x) for x in X]))
    i+=1
    
    # break
axs[0].set_xlabel('')
axs[0].set_ylabel('$n (mm^{-2})$')

axs[1].set_xlabel('position ($mm$)')
axs[1].set_ylabel(r'$\rho (mm.mm^{-2})$')

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
fix.savefig(os.path.join(PLOTS_DIR, 'cylindrical_drift_diffusion.png'), dpi=150, bbox_inches='tight')
print(f'Plot saved to {PLOTS_DIR}')

# ── Save simulation output for surrogate modelling ────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'CylindricalDrift')
os.makedirs(DATA_DIR, exist_ok=True)

np.save(os.path.join(DATA_DIR, 'n_field.npy'),   density_tip)   # shape [nT/keep, nx]
np.save(os.path.join(DATA_DIR, 'rho_field.npy'), density_in)    # shape [nT/keep, nx]
np.save(os.path.join(DATA_DIR, 'x.npy'),         X)             # shape [nx]
np.save(os.path.join(DATA_DIR, 't.npy'),         np.linspace(0, T, int(nT / keep)))  # shape [nT/keep]
print(f'Data saved to {DATA_DIR}')