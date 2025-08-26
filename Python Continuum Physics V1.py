from re import S
from sys import float_repr_style
import numpy as np
import scipy.sparse as spar
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from typing import Union


inner = ( slice(1,-1), slice(1,-1) )

skip = ( slice(None, None, 2), slice(None, None, 2) )

b_order = [ 'x_min', 'x_max', 'y_min', 'y_max' ]

b_dict = { 'dr': 0, 'nm': 1 }




class grid:
    def __init__(self,
                 x_min: float,
                 x_max: float,
                 y_min: float,
                 y_max: float,
                 bins_x: int,
                 bins_y:int
                 ):
        self.bins_x = bins_x
        self.bins_y = bins_y
        xvals = np.linspace( x_min, x_max, bins_x + 1 )
        yvals = np.linspace( y_min, y_max, bins_y + 1 )
        self.X, self.Y = np.meshgrid( xvals, yvals, indexing='ij' )

        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        self.length_x = abs( x_max - x_min )
        self.length_y = abs( y_max - y_min )

        self.dx = self.length_x / bins_x
        self.dy = self.length_y / bins_y

        ## -- Build Operator Matrices...

        ## -- Fixed Boundary Values (Dirichlet)

        self.Identity_x = spar.identity( bins_x - 1 )
        self.Identity_y = spar.identity( bins_y - 1 )
        self.Identity_prod = spar.identity( ( bins_x - 1 )*( bins_y - 1 ) )

        ones_x = np.ones( bins_x - 1 )
        ones_y = np.ones( bins_y - 1 )
        ones_prod = np.ones( ( bins_x - 1 )*( bins_y - 1 ) )

        Px_dir = spar.diags( [ ones_x[1:], -ones_x[1:] ], [1,-1] ) * 0.5 / self.dx
        Py_dir = spar.diags( [ ones_y[1:], -ones_y[1:] ], [1,-1] ) * 0.5 / self.dy

        Pxx_dir = spar.diags( [ -2*ones_x, ones_x[1:], ones_x[1:] ],
                              [0,1,-1]
                             ) / self.dx**2

        Pyy_dir = spar.diags( [ -2*ones_y, ones_y[1:], ones_y[1:] ],
                              [0,1,-1]
                             ) / self.dy**2

        self.Dx, self.Dy = spar.kron( Px_dir, self.Identity_y ), spar.kron( self.Identity_x, Py_dir )
        self.Dxx, self.Dyy = spar.kron( Pxx_dir, self.Identity_y ), spar.kron( self.Identity_x, Pyy_dir )



        # - 'insulator' matrices which can be added the the bare (vanishing dirichlet) matrices

        insulators_x = [
            spar.diags( [ [-1] + [0] * ( bins_x - 2 ) ] , [0] ) * 0.5/self.dx,
            spar.diags( [ [0] * ( bins_x - 2 ) + [1] ] , [0] ) * 0.5/self.dx,
            spar.diags( [ [1] + [0] * ( bins_x - 2 ) ] , [0] ) / self.dx**2,
            spar.diags( [ [0] * ( bins_x - 2 ) + [1] ] , [0] ) / self.dx**2
            ]

        self.insulators_x = [ spar.kron( ins , self.Identity_y ) for ins in insulators_x ]

        insulators_y = [
            spar.diags( [ [-1] + [0] * ( bins_y - 2 ) ] , [0] ) * 0.5/self.dy,
            spar.diags( [ [0] * ( bins_y - 2 ) + [1] ] , [0] ) * 0.5/self.dy,
            spar.diags( [ [1] + [0] * ( bins_y - 2 ) ] , [0] ) / self.dy**2,
            spar.diags( [ [0] * ( bins_y - 2 ) + [1] ] , [0] ) / self.dy**2,
            ]

        self.insulators_y = [ spar.kron( self.Identity_x, ins ) for ins in insulators_y  ]



        ## -- Insulated Boundary (Vanishing Neumann)

        self.Dx_ins = self.Dx + sum( self.insulators_x[:2] )
        self.Dxx_ins = self.Dxx + sum( self.insulators_x[2:] )

        self.Dy_ins = self.Dy + sum( self.insulators_y[:2] )
        self.Dyy_ins = self.Dyy + sum( self.insulators_y[2:] )

        ## -- Laplacian Operators

        self.Laplacian_dir = self.Dxx + self.Dyy
        self.Laplacian_ins = self.Dxx_ins + self.Dyy_ins


        # self.Laplacian_ins_ext = spar.csr_matrix( spar.block_array( [[ self.Laplacian_ins, ones_prod[:,None] ],
        #                                                               [ ones_prod[None,:], None ]] ) )


        self.Laplacian_ins_ext = self.extend_op( self.Laplacian_ins )


        pass

    def extend_op(self, op):
            shp = op.shape
            mat = np.ones(  ( shp[0]+1, shp[1] + 1 )  )
            mat[:-1, :-1] = op.toarray()
            mat[-1, -1] = 0
            return spar.csr_matrix( mat )





class scalar_field:
    def __init__(self,
                 coords: grid,
                 field_init: np.ndarray,
                 boundary_type: dict
                 ):
        self.coords = coords
        self.field_init = field_init
        self.boundary_type = boundary_type

        self.field_state = field_init[inner].flatten()


        zerosmall = np.zeros_like( coords.X[inner], dtype=self.field_init.dtype )

        bvals = [
            field_init[0, 1:-1],
            field_init[-1, 1:-1],
            field_init[1:-1, 0],
            field_init[1:-1, -1]
            ]

        bvals_nm = [
            ( field_init[1, 1:-1] - field_init[0, 1:-1] ),
            ( field_init[-1, 1:-1] - field_init[-2, 1:-1] ),
            ( field_init[1:-1, 1] - field_init[1:-1, 0] ),
            ( field_init[1:-1, -1] - field_init[1:-1, -2] )
            ]

        
        self.b_binary = [ b_dict[ self.boundary_type[bd] ] for bd in b_order ]
        
        sc = self.b_binary

        bvecs = [
            ( 1 - sc[0] ) * bvals[0] - sc[0] * bvals_nm[0],
            ( 1 - sc[1] ) * bvals[1] + sc[1] * bvals_nm[1],
            ( 1 - sc[2] ) * bvals[2] - sc[2] * bvals_nm[2],
            ( 1 - sc[3] ) * bvals[3] + sc[3] * bvals_nm[3]
            ]


        # self.b_binary = [ b_dict[ self.boundary_type[bd] ] for bd in b_order ]

        # self.b_binary_inv = [ 1 - bt for bt in self.b_binary ]

        # self.boundary_values = [
        #     np.append( bvals[0][None,:], zerosmall[:-1], axis=0 ).flatten() * self.b_binary_inv[0],
        #     np.append( zerosmall[:-1], bvals[1][None,:], axis=0 ).flatten() * self.b_binary_inv[1],
        #     np.append( bvals[2][:,None], zerosmall[:,:-1], axis=1 ).flatten() * self.b_binary_inv[2],
        #     np.append( zerosmall[:,:-1], bvals[3][:,None], axis=1 ).flatten() * self.b_binary_inv[3]
        #     ]

        # self.boundary_values_nm = [
        #     np.append( bvals_nm[0][None,:], zerosmall[:-1], axis=0 ).flatten() * self.b_binary[0],
        #     np.append( zerosmall[:-1], bvals_nm[1][None,:], axis=0 ).flatten() * self.b_binary[1],
        #     np.append( bvals_nm[2][:,None], zerosmall[:,:-1], axis=1 ).flatten() * self.b_binary[2],
        #     np.append( zerosmall[:,:-1], bvals_nm[3][:,None], axis=1 ).flatten() * self.b_binary[3]
        #     ]


        self.boundary_values = [
            np.append(  bvecs[0][None,:] , zerosmall[:-1] , axis=0  ).flatten(),
            np.append(  zerosmall[:-1] , bvecs[1][None,:] , axis=0  ).flatten(),
            np.append(  bvecs[2][:,None] , zerosmall[:,:-1] , axis=1  ).flatten(),
            np.append(  zerosmall[:,:-1] , bvecs[3][:,None] , axis=1  ).flatten()
            ]

        self.make_Dbounds()
        
        self.make_derivative_ops()

        pass


    def make_Dbounds(self):
        b = self.boundary_values
        dx, dy = self.coords.dx, self.coords.dy
        out_dict = {
            'Dx' : ( b[1] - b[0] ) * 0.5/dx,
            'Dxx': ( b[1] + b[0] ) / dx**2,
            'Dy' : ( b[3] - b[2] ) * 0.5/dy,
            'Dyy': ( b[3] + b[2] ) / dy**2,
        }

        self.Dbounds = out_dict

        return()


    def make_derivative_ops(self):
        b = self.b_binary
        self.Dx = self.coords.Dx + b[0] * self.coords.insulators_x[0] + b[1] * self.coords.insulators_x[1]
        self.Dxx = self.coords.Dxx + b[0] * self.coords.insulators_x[2] + b[1] * self.coords.insulators_x[3]

        self.Dy = self.coords.Dy + b[2] * self.coords.insulators_y[0] + b[3] * self.coords.insulators_y[1]
        self.Dyy = self.coords.Dyy + b[2] * self.coords.insulators_y[2] + b[3] * self.coords.insulators_y[3]

        self.Laplacian = self.Dxx + self.Dyy

        return()

    def matrixform(self):
        mat = np.zeros_like( self.coords.X, dtype = self.field_init.dtype )
        mat[inner] = self.field_state.reshape( self.coords.X[inner].shape )

        b = self.b_binary
        f0 = self.field_init

        mat[0] = ( 1 - b[0] ) * f0[0] + b[0] * mat[1]
        mat[-1] = ( 1 - b[1] ) * f0[-1] + b[1] * mat[-2]
        mat[:,0] = ( 1 - b[2] ) * f0[:,0] + b[2] * mat[:,1]
        mat[:,-1] = ( 1 - b[3] ) * f0[:,-1] + b[3] * mat[:,-2]

        return mat





class solid:
    def __init__(self,
                 coords: grid,
                 temperature_init: np.ndarray,
                 temperature_source: np.ndarray,
                 heat_const: float,
                 boundary_type = list[int]
                 ):
        self.coords = coords

        self.St_state = temperature_source[inner].flatten()

        self.heat_const = heat_const

        self.temp_field = scalar_field( coords, temperature_init, boundary_type )

        pass


    def step_forward(self,
                     dt: float
                     ):
        T = self.temp_field.field_state
        b = self.temp_field.Dbounds
        sys_matrix = self.coords.Identity_prod - dt * self.heat_const * self.temp_field.Laplacian
        
        T += dt * (  self.St_state + self.heat_const * ( b['Dxx'] + b['Dyy'] )  )

        T = spar.linalg.spsolve( sys_matrix, T )

        self.temp_field.field_state = T

        return()


    def plot_temp(self,
                  ax: plt.axis,
                  Tmin: float,
                  Tmax: float
                  ):
        hplot = ax.pcolormesh(
            self.coords.X,
            self.coords.Y,
            self.temp_field.matrixform(),
            cmap='jet',
            vmin=Tmin,
            vmax=Tmax
            )

        return hplot

    def plot_all(self,
                 ax: plt.axis,
                 Tmin: float,
                 Tmax: float
                 ):
        return [ self.plot_temp( ax, Tmin, Tmax ) ]

 


class fluid:
    def __init__(self,
                 coords: grid,
                 flow_x_init: np.ndarray,
                 flow_y_init: np.ndarray,
                 temp_init: np.ndarray,
                 flow_x_source: np.ndarray,
                 flow_y_source: np.ndarray,
                 temp_source: np.ndarray,
                 boundary_type_x: list[int],
                 boundary_type_y: list[int],
                 boundary_type_temp: list[int],
                 density: float,
                 viscosity: float,
                 heat_const: float
                 ):
        self.coords = coords
        
        self.flow_x = scalar_field( coords, flow_x_init, boundary_type_x )
        self.flow_y = scalar_field( coords, flow_y_init, boundary_type_y )
        
        self.temp = scalar_field( coords, temp_init, boundary_type_temp )

        self.Sx_state = flow_x_source[inner].flatten()
        self.Sy_state = flow_y_source[inner].flatten()
        self.St_state = temp_source[inner].flatten()

        self.density = density
        self.viscosity = viscosity
        self.heat_const = heat_const

        pass


    def step_forward(self,
                     dt: float
                     ):
        u,v = self.flow_x.field_state, self.flow_y.field_state
        uu, vv = spar.diags( [u], [0] ), spar.diags( [v], [0] )

        rhs_matrix_x = (
            - uu @ self.flow_x.Dx
            - vv @ self.flow_x.Dy
            + self.viscosity / self.density * self.flow_x.Laplacian
            )

        rhs_matrix_y = (
            - uu @ self.flow_y.Dx
            - vv @ self.flow_y.Dy
            + self.viscosity / self.density * self.flow_y.Laplacian
            )

        sys_matrix_x = self.coords.Identity_prod - dt * rhs_matrix_x
        sys_matrix_y = self.coords.Identity_prod - dt * rhs_matrix_y

        Dbx = self.flow_x.Dbounds
        Dby = self.flow_y.Dbounds

        bx = - uu @ Dbx['Dx'] - vv @ Dbx['Dy'] + self.viscosity / self.density * ( Dbx['Dxx'] + Dbx['Dyy'] )
        by = - uu @ Dby['Dx'] - vv @ Dby['Dy'] + self.viscosity / self.density * ( Dbx['Dxx'] + Dbx['Dyy'] )

        u += dt * ( self.Sx_state + bx )
        v += dt * ( self.Sy_state + by )

        u = spar.linalg.spsolve( sys_matrix_x, u )
        v = spar.linalg.spsolve( sys_matrix_y, v )

        div = np.append( self.flow_x.Dx @ u + self.flow_y.Dy @ v, 0 )

        p = spar.linalg.spsolve( self.coords.Laplacian_ins_ext, div )[:-1]

        u -= self.coords.Dx_ins @ p
        v -= self.coords.Dy_ins @ p

        self.flow_x.field_state = u
        self.flow_y.field_state = v

        T = self.temp.field_state
        
        rhs_matrix_temp = (
            - uu @ self.temp.Dx
            - vv @ self.temp.Dy
            + self.heat_const / self.density * self.temp.Laplacian
            )

        sys_matrix_temp = self.coords.Identity_prod - dt * rhs_matrix_temp

        Dbt = self.temp.Dbounds

        bt = - uu @ Dbt['Dx'] - vv @ Dbt['Dy'] + self.heat_const / self.density * ( Dbt['Dxx'] + Dbt['Dyy'] )

        T += dt * ( self.St_state + bt )

        T = spar.linalg.spsolve( sys_matrix_temp, T )

        self.temp.field_state = T

        return()

    def plot_temp(self,
                 ax: plt.axis,
                 Tmin: float,
                 Tmax: float
                 ):
        hplot = ax.pcolormesh(
            self.coords.X,
            self.coords.Y,
            self.temp.matrixform(),
            cmap='jet',
            vmin = Tmin,
            vmax = Tmax
            )

        return hplot

    def plot_flow(self,
                  ax: plt.axis
                  ):
        qplot = ax.quiver(
            self.coords.X[skip],
            self.coords.Y[skip],
            self.flow_x.matrixform()[skip],
            self.flow_y.matrixform()[skip],
            pivot='middle'
            )

        return qplot

    def plot_all(self,
                 ax: plt.axis,
                 Tmin: float,
                 Tmax:float
                 ):
        hplot = self.plot_temp( ax, Tmin, Tmax )
        qplot = self.plot_flow( ax )
        
        return [ hplot, qplot ]






class wavefunction:
    def __init__(self,
                 coords: grid,
                 psi_init: np.ndarray,
                 potential: np.ndarray,
                 boundary_type: dict,
                 mass: float,
                 hbar: float
                 ):
        self.coords = coords
        self.potential = potential
        self.mass = mass
        self.hbar = hbar

        self.wavefun = scalar_field( coords, psi_init, boundary_type )

        self.Clap = 0.5j * self.hbar / self.mass

        self.pot_state = self.potential[inner].flatten()

        pass

    def step_forward(self, dt):
        psi = self.wavefun.field_state

        V_diag = spar.diags( [self.pot_state], [0] )
        rhs_matrix = self.Clap * self.wavefun.Laplacian - 1.0j * self.hbar * V_diag
        sys_matrix = self.coords.Identity_prod - dt * rhs_matrix

        Db = self.wavefun.Dbounds

        psi += dt * self.Clap * ( Db['Dxx'] + Db['Dyy'] )

        psi = spar.linalg.spsolve( sys_matrix, psi )

        self.wavefun.field_state = psi
        
        return()


    def plot_all(self,
                 ax: plt.axis,
                 min_val: float,
                 max_val: float
                 ):
        pd_matrix = abs( self.wavefun.matrixform() )**2
        pd_plot = ax.pcolormesh(
            self.coords.X,
            self.coords.Y,
            pd_matrix,
            cmap = 'jet',
            vmin = min_val,
            vmax = max_val
            )

        return [ pd_plot ]







class physics_sim:
    def __init__(self,
                 physics_comp: Union[fluid, solid],
                 t_min: float,
                 t_max: float,
                 dt: float,
                 figure_size: list[float],
                 dpi: int,
                 sim_title: str,
                 gif_title: str
                 ):
        self.physics_comp = physics_comp
        
        self.t_min = t_min
        self.t_max = t_max
        self.dt = dt

        self.t_values = np.arange( t_min, t_max + dt, step=dt )

        self.fig, self.ax = plt.subplots( figsize = figure_size, dpi = dpi )

        self.dpi = dpi

        self.sim_title = sim_title
        self.gif_title = gif_title + '.gif'

        self.fig.suptitle( self.sim_title )

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_aspect('equal')

        self.physics_plots = self.physics_comp.plot_all( self.ax, 0, 1 )

        self.writer = PillowWriter( fps=10, metadata=None )

        pass


    def run_simulation(self):
        maxcount = 0.1/dt
        count = maxcount

        with self.writer.saving( self.fig, self.gif_title, self.dpi ):
            for t in self.t_values:
                if count+1 < maxcount:
                    count += 1
                else:
                    count = 0

                    for plots in self.physics_plots:
                        plots.remove()

                    self.physics_plots = self.physics_comp.plot_all( self.ax, 0, 1 )

                    self.writer.grab_frame()

                    pass

                self.physics_comp.step_forward( self.dt )

        
        self.writer.finish()

        plt.show()

        return()










## -------------------------|
## -- Simulation Setup
##--------------------------|



tmax = 4
dt = 0.01
tvals = np.arange( 0, tmax + dt, step=dt )


coords = grid( 0, 1, 0, 1, 100, 100 )

X, Y = coords.X, coords.Y

u0 = v0 = Source_x = np.zeros_like(X)

from numpy import complex128, pi

# T0 =  0.5 * ( 1 + np.cos( pi * Y / coords.length_y ) )


T0 = 0.5 * ( 1 - np.cos( 2*pi * X / coords.length_x ) ) * ( 1 - Y / coords.length_y )


Source_y = (
    np.sin( 2*pi * Y / coords.length_y ) * np.heaviside( coords.length_y / 2 - Y, 0 ) *
    1.2 * ( 1 - np.cos( 2*pi * X / coords.length_x ) ) * np.exp( -2*X / coords.length_x )
    ) * 0.7


# Source_temp = (
#     np.sin( 2*np.pi * Y / coords.length_y ) * np.heaviside( Y - coords.length_y / 2, 0 ) *
#     np.sin( 2*np.pi * X / coords.length_x ) * np.heaviside( X - coords.length_x / 2, 0 )
#     ) * 0.3

Source_temp = np.zeros_like(X)


# - Fluid Parameters
rho = 2
mu = 2e-3
k = 0.01
beta = 0.0


## -- Method to construct "boundary_type" dictionaries
def make_bt( xmin, xmax, ymin, ymax ):
    return {
        'x_min': xmin,
        'x_max': xmax,
        'y_min': ymin,
        'y_max': ymax
        }





# fluid_1 = fluid( coords,
#                 u0, v0, T0,
#                 Source_x, Source_y, Source_temp,
#                 make_bt( 'dr', 'dr', 'nm', 'nm' ),
#                 make_bt( 'nm', 'nm', 'dr', 'dr' ),
#                 make_bt( 'nm', 'nm', 'dr', 'dr' ),
#                 rho, mu, k
#                 )


# fluid_sim_1 = physics_sim( fluid_1, 0, tmax, dt, [10,8], 70,
#                           'Heat distribution in a driven fluid -- Horizontally Insulated, Vertically Fixed',
#                           'FSV4.1 Driven Flow - Mixed Boundaries'
#                           )

# fluid_sim_1.run_simulation()





# solid_1 = solid(
#     coords,
#     T0,
#     Source_temp,
#     k,
#     make_bt( 'nm', 'nm', 'dr', 'dr' )
#     )


# solid_sim_1 = physics_sim( solid_1, 0, tmax, dt, [10,8], 70,
#                           'Heat distribution in a solid conductor',
#                           'PSV4.1 Solid Conductor Example 1'
#                          )

# solid_sim_1.run_simulation()




psi_0 = np.sin( pi * X / coords.length_x ) * np.sin( pi * Y / coords.length_y ) * (1+0j)

# def V_fun(x,y):
#     return 0.5 * ( x**2 + y**2 )

# V_mat = V_fun( X, Y )

V_mat = np.zeros_like(X)


wfun = wavefunction(
    coords,
    psi_0,
    V_mat,
    make_bt( 'dr', 'dr', 'dr', 'dr' ),
    1,
    1
    )


quantum_sim = physics_sim( wfun, 0, tmax, dt, [10,8], 70,
                          'Plot of Probability Density',
                          'PSV4.1 Wavefunction Example 1'
                         )


quantum_sim.run_simulation()