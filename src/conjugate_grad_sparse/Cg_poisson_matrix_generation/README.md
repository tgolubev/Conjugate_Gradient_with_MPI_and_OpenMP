#  Poisson matrix generation from 3D Drift Diffusion for Holes with Finite Differences Model

  This is a portion of my research code which generates a 3D Poisson matrix which describes a real physical system. I am using this to test my conjugate gradient algorithm with a realistic matrix.

## Boundary conditions for Poisson equation are:

* -a fixed voltage at (x,0) and (x, Nz) defined by V_bottomBC and V_topBC which are defining the  electrodes
*  -insulating boundary conditions: V(0,y,z) = V(1,y,z) and V(N+1,y,z) = V(N,y,z) (N is the last INTERIOR mesh point).So the potential at the boundary is assumed to be the same as just inside
 the boundary. Gradient of potential normal to these boundaries is 0.
V(x,0,z) = V(x,1,z) and V(x,N+1,z) = V(x,N,z)

* Matrix equation is AV*V = bV is sparse matrices (generated using spdiag), for the Poisson equations.
* V is the solution for electric potential
* bV is the rhs of Poisson eqn which contains the charge densities and boundary conditions
