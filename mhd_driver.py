import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from flux_conservative import lax_wendroff as lw

def mhd_flux(qin):
    q = qin.T

    r = q[0]  # density
    px = q[1] # x-momentum
    py = q[2] # y-momentum
    pz = q[3] # z-momentum
    Bx = q[4] # x component of magnetic field
    By = q[5] # y component of magnetic field
    Bz = q[6] # z component of magnetic field
    w = q[7]  # energy

    P = 4/3 * w - 2/(3*r) * (px**2+py**2+pz**2) + 1/3 * (Bx**2+By**2+Bz**2)
    ret = np.empty_like(q)

    ret[0] = px
    ret[1] = 1/r * px**2 + 0.5*P - Bx**2
    ret[2] = 1/r * px*py - Bx*By
    ret[3] = 1/r * px*pz - Bx*Bz
    ret[4] = 0
    ret[5] = -1/r * (Bx*py - px*By)
    ret[6] = -1/r * (Bx*pz - px*Bz)
    ret[7] = 1/r * ((w+0.5*P)*px - (px*Bx+py*By+pz*Bz)*Bx)

    return ret.T

one   = lambda x: 1.
zero  = lambda x: 0.
gauss = lambda x: 0.1*np.exp(-(x-50)**2/8)
w_init = lambda x: 0.5*gauss(x)**2/one(x) + 2.5*one(x) # includes some pressure

mhd_init = [one,    # density
            zero,   # x-momentum
            gauss,  # y-momentum
            zero,   # z-momentum
            one,    # Bx
            zero,   # By
            zero,   # Bz
            w_init] # energy
            
fig, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.set_title('$u_x$')
ax2.set_title('$B_x$')
ax3.set_title('$\\rho$')
ax3.set_ylim(.9,1.1)

lines = []
fd = lw(100, .005, mhd_init, mhd_flux, m=1000)
while(fd.t < 100):
    u_line,   = ax1.plot(fd.grid, fd.q[:,2]/fd.q[:,0], color='black')
    By_line,  = ax2.plot(fd.grid, fd.q[:,5], color='green')
    rho_line, = ax3.plot(fd.grid, fd.q[:,0], color='blue')
    lines.append([u_line, By_line, rho_line])
    fd.step(200)

ani = animation.ArtistAnimation(fig, lines)

plt.show()
