# Local angle and crystallinity

Compute the local angle and crystallinity of a set of points. 
The nearest neighbours are identify using a threshold (KDTree in Scipy).
The local angle and cristallinity of an atom $i$ are computed from the local director field (see https://en.wikipedia.org/wiki/Hexatic_phase and SI of https://www.nature.com/articles/s41567-021-01429-3):
$$\psi_i= \frac{1}{N_i} \sum_j^{\langle i \rangle} \exp(i N_\mathrm{ord} \theta_{ij}), $$
where $\theta_{ij}$ is the angle of the bond vector $\delta_{ij}=r_j-r_i$, $N_i$ is the number of nearest neighbour of $i$, and $N_\mathrm{ord}$ is the symmetry order of the lattice (6 for triangular, 4 for square).

The local angle is the computed as the argument of the complex function $\psi$.

The crystallinity as the absolute value of function $\psi$.

These snippets are implemented in the script ```local_angle-complex.py```. You can use it with 
```
./local_angel-complex.py test/test.xyz 1.3856406460551018  60 6  > test/test-langle.xyz
```

#### Requirment

For python function
- Scipy

For xyz bash script
- ASE

For Jupyter notebook
- matplotlib
