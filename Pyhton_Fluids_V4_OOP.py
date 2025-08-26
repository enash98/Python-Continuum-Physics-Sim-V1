import numpy as np
import scipy.sparse as spar
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


inner = ( slice(1,-1,None), slice(1,-1,None) )

skip = ( slice(None, None, 2), slice(None, None, 2) )


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

        self.Laplacian_ins_ext = spar.csr_matrix( spar.block_array( [[ self.Laplacian_ins, ones_prod[:,None] ],
                                                                      [ ones_prod[None,:], None ]] ) )

        pass



       


class fluid:
    def __init__(self,
                 coords: grid,
                 velocity_x_init: np.ndarray,       # - must have the same 'shape' as coords
                 velocity_y_init: np.ndarray,       # - (see above)
                 temperature_init: np.ndarray,      # - (see above)
                 flow_source_x: float,
                 flow_source_y: float,
                 temperature_source: float,
                 density: float,
                 viscosity: float,
                 heat_const: float,
                 beta: float
                 ):
        self.coords = coords
        
        self.density = density
        self.viscosity = viscosity
        self.heat_const = heat_const

        # source terms
        self.Sx_state = flow_source_x[inner].flatten()
        self.Sy_state = flow_source_y[inner].flatten()
        self.St_state = temperature_source[inner].flatten()

        # velocity state
        self.velocity_x_init = velocity_x_init
        self.velocity_y_init = velocity_y_init

        self.velocity_x_state = velocity_x_init[inner].flatten()
        self.velocity_y_state = velocity_y_init[inner].flatten()

        # temperature state
        self.temperature_init = temperature_init
        self.temperature_state = temperature_init[inner].flatten()

        self.beta = beta

        pass


    def step_forward(self, dt: int):
        u,v,T = self.velocity_x_state, self.velocity_y_state, self.temperature_state
        
        u_diag, v_diag = spar.diags( [u], [0] ), spar.diags( [v], [0] )

        rhs_matrix = - u_diag @ self.coords.Dx - v_diag @ self.coords.Dy + self.viscosity / self.density * self.coords.Laplacian_dir
        sys_matrix = self.coords.Identity_prod - dt * rhs_matrix

        convect = - self.beta * self.coords.Dy_ins @ T

        u = spar.linalg.spsolve( sys_matrix, u + dt * self.Sx_state )
        v = spar.linalg.spsolve( sys_matrix, v + dt * ( self.Sy_state + convect ) )

        # - Solve for 'pressure' and project...

        div = np.append( self.coords.Dx @ u + self.coords.Dy @ v, 0 )
        press_state = spar.linalg.spsolve( self.coords.Laplacian_ins_ext, div )[:-1]

        u -= self.coords.Dx_ins @ press_state
        v -= self.coords.Dy_ins @ press_state

        self.velocity_x_state = u
        self.velocity_y_state = v

        # - Evlove temperature...

        rhs_matrix = - u_diag @ self.coords.Dx_ins - v_diag @ self.coords.Dy_ins + self.heat_const / self.density * self.coords.Laplacian_ins
        sys_matrix = self.coords.Identity_prod - dt * rhs_matrix

        T = spar.linalg.spsolve( sys_matrix, T + dt * self.St_state )

        self.temperature_state = T


        return()

    def flow_matrixform(self):
        matx = np.zeros_like( self.coords.X )
        maty = np.zeros_like( self.coords.X )

        matx[inner] = self.velocity_x_state.reshape( self.coords.X[inner].shape )
        maty[inner] = self.velocity_y_state.reshape( self.coords.X[inner].shape )

        return matx, maty

    def temp_matrixform(self):
        mat = np.zeros_like( self.coords.X )
        mat[inner] = self.temperature_state.reshape( self.coords.X[inner].shape )

        mat[0], mat[-1], mat[:,0], mat[:,-1] = mat[1], mat[-2], mat[:,1], mat[:,-2]

        return mat

    def plot_flow( self, ax ):
        u_mat, v_mat = self.flow_matrixform()
        qplot = ax.quiver(
            self.coords.X[skip],
            self.coords.Y[skip],
            u_mat[skip],
            v_mat[skip],
            pivot = 'middle'
            )
        return qplot

    def plot_temp( self, ax, Tmin, Tmax ):
        T_mat = self.temp_matrixform()
        hmap = ax.pcolormesh(
            self.coords.X,
            self.coords.Y,
            T_mat,
            cmap = 'jet',
            vmin = Tmin,
            vmax = Tmax
            )

        return hmap





class fluid_sim:
    def __init__(self,
                 fluid: fluid,
                 t_min: float,
                 t_max: float,
                 dt: float,
                 figure_size: list[float],
                 dpi: int
                 ):
        self.fluid = fluid
        
        self.t_min = t_min
        self.t_max = t_max
        self.dt = dt

        self.t_values = np.arange( t_min, t_max + dt, step=dt )

        self.fig, self.ax = plt.subplots( figsize = figure_size, dpi = dpi )

        self.dpi = dpi

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_aspect('equal')

        self.heat_plot = self.ax.pcolormesh( X, Y, fluid.temperature_init, cmap = 'jet', vmin=0, vmax=1 )

        self.colour_bar = self.fig.colorbar( self.heat_plot, ax = self.ax )

        self.quiver_plot = self.ax.quiver( self.fluid.coords.X, self.fluid.coords.Y,
                                           self.fluid.velocity_x_init, self.fluid.velocity_y_init,
                                           pivot = 'middle'
                                          )

        self.writer = PillowWriter( fps = 10, metadata = None )

        pass

    def run_simulation(self):
        maxcount = 0.1/dt
        count = maxcount

        with self.writer.saving( self.fig, 'fluid simulation V4.gif', self.dpi ):
            for t in self.t_values:
                if count+1 < maxcount:
                    count += 1
                else:
                    count = 0

                    self.quiver_plot.remove()
                    self.heat_plot.remove()

                    self.quiver_plot = self.fluid.plot_temp( self.ax, 0, 1 )

                    self.heat_plot = self.fluid.plot_flow( self.ax )

                    self.writer.grab_frame()

                    pass

                self.fluid.step_forward( self.dt )

        
        self.writer.finish()

        plt.show()

        return()




## -------------------------|
## -- Simulation Setup
##--------------------------|



tmax = 10
dt = 0.01
tvals = np.arange( 0, tmax + dt, step=dt )


coords = grid( 0, 1, 0, 1, 100, 100 )

X, Y = coords.X, coords.Y

u0 = v0 = T0 = Source_x = np.zeros_like(X)

# T0 = (
#     0.5 * ( 1 - np.cos( 2 * np.pi * X / coords.length_x ) ) *
#     0.5 * ( 1 - np.cos( 2 * np.pi * Y / coords.length_y ) )
#     )


Source_y = (
    np.sin( 2*np.pi * Y / coords.length_y ) * np.heaviside( coords.length_y / 2 - Y, 0 ) *
    np.sin( 2*np.pi * X / coords.length_x ) * np.heaviside( coords.length_x / 2 - X, 0 )
    ) * 0.5

Source_temp = (
    np.sin( 2*np.pi * Y / coords.length_y ) * np.heaviside( Y - coords.length_y / 2, 0 ) *
    np.sin( 2*np.pi * X / coords.length_x ) * np.heaviside( X - coords.length_x / 2, 0 )
    ) * 0.25

# - Fluid Parameters
rho = 2
mu = 3e-3
k = 5e-3
beta = 0.0


# Build Fluid

fluid_1 = fluid( coords,
                u0, v0, T0,
                Source_x, Source_y, Source_temp,
                rho, mu, k , beta
               )




fluid_sim_1 = fluid_sim( fluid_1, 0, tmax, dt, [10,8], 70 )

fluid_sim_1.run_simulation()


