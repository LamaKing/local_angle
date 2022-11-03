#!/usr/bin/env python3

import sys, logging
import numpy as np
from time import time
from scipy.spatial import cKDTree
from numpy import pi
import ase, ase.io

def local_angle(pos, d0, th_period, Nord, debug=False, quiet=False):
    """Local angle with respect x with nearest neighbour in set of points.

    The angle of the atom i is defined using the orientational order:
        psi_i= 1/N_i sum_j^{<i>} exp(i Nord theta_ij),
    where the extends over the nearest neighbors of i, <i>, Nord is the symmetry order (6 for triangular symmetry, 4 for square).
    The angle theta_ij is computed from the i->j bond vector delta: theta_ij = arctan(delta_y/delta_x) % th_period. The period is given in degrees.
    Nearest neighbour are identified using the given distance d0 using scipy KDTree.

    The script returns the local angle of each atom arg(psi) and the crystalline order |psi|.
    """

    c_log = logging.getLogger(__name__)
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    logging.basicConfig(format='[%(levelname)5s - %(funcName)10s] %(message)s')
    c_log.setLevel(logging.INFO)
    if debug: c_log.setLevel(logging.DEBUG)
    if quiet: c_log.setLevel(logging.WARNING)
    progress_out = 4 # update this many times during computation

    t0 = time() # start the clock

    neigh_tree = cKDTree(pos, copy_data=True) # Neighour list object
    nn = neigh_tree.query_pairs(d0, output_type='ndarray') # Nearest neighbours mask indices
    pos_nn = pos[nn] # Position of nearest neighbours

    psi = np.zeros(pos.shape[0], dtype=np.complex128) # init vector, as complex (or imag part is discarded).
    for cind in range(pos.shape[0]): # loop on atoms
        if cind % int(pos.shape[0]/progress_out) == 0: c_log.info("At %.0f%%" % (cind/pos.shape[0]*100))
        mask_cnn = np.logical_or(nn[:,0] == cind, nn[:,1] == cind) # select the nearest neighbour of each atom.
        for il in pos_nn[mask_cnn]: # loop on nn of current atom
            dd = il[1] - il[0] # !!! non ordered pairs (relevant for plotting)
            c_angle = np.round(np.arctan2(dd[1], dd[0]), decimals=6) % (th_period*pi/180) # rounding help near period
            psi[cind] += np.exp(1j*Nord*c_angle)
        psi[cind] /= pos_nn[mask_cnn].shape[0] # average over NN

    # average angle from complex function
    avg_angle = 180/pi*np.angle(psi)/Nord
    # crystallinity as absolute value of complex function
    avg_crystal = np.abs(psi)

    texec = time() - t0 # stop the clock
    c_log.info("Done in %is" % texec)
    return avg_angle, avg_crystal

if __name__ == '__main__':
    """Read xyz trajectory and compute local angle and crystallinity.

    Arguments are xyz_filename nn_cutoff theta_period symmetry_order (frame_skip=1)

    Prints xyz with local angle in as vx and crystallinity as vy.
    """
    t0 = time()

    c_log = logging.getLogger(__name__) # Set name of the function
    # Adopted format: level - current function name - message. Width is fixed as visual aid.
    logging.basicConfig(format='[%(levelname)5s - %(funcName)10s] %(message)s')
    c_log.setLevel(logging.INFO)

    progress_out = 10

    xyz_fname = sys.argv[1] # xyz trajectory
    d0 = float(sys.argv[2]) # cutoff dist for NN
    th_period = float(sys.argv[3]) # angular period
    Nord = float(sys.argv[4]) # symmetry order
    nskip = 1 # read every this many frames
    if len(sys.argv)>5: nskip = int(sys.argv[5])

    c_log.info('Cutoff NN d0=%.5g' % (d0))
    c_log.info('Load traj %s (nskip=%i)' % (xyz_fname, nskip))

    # This might not be too smart, reading it all at once
    traj = ase.io.read(xyz_fname, format='xyz', index=slice(0,-1,nskip))
    if len(traj) == 0:
        traj = [ase.io.read(xyz_fname, format='xyz')]
    c_log.info('Loaded N=%i' % (len(traj)))

    for i, frame in enumerate(traj):
        N = len(frame)
        if i % int(len(traj)/progress_out+1) == 0: c_log.info('On %.0f%%' % (100*i/len(traj)))

        pos = frame.positions
        avg_angle, avg_crystal = local_angle(pos, d0=d0, th_period=th_period, Nord=Nord, quiet=True) # do not speak
        field = np.vstack((avg_angle, avg_crystal, np.zeros(N))).T
        frame.set_momenta(field) # CAREFUL WITH ASE CONVENTION (e.g. multiplying for the mass if you use set_velocity beacuse it writes momenta in xyz...)

    ase.io.write(sys.stdout, traj, format='extxyz')

    texec = time()-t0
    c_log.info("Done in %is (%.2fmin)" % (texec, texec/60))
