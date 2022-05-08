"""
Methods and classes which are no longer used, but exist for both
backwards compatibility and as a record of computation which once existed.

This file contains all code that is considered to be old from any of the
modules in `computation`, sorted by comments differentiating their purpose.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


__all__ = ['ode_system_from_matrix', 'plot_autonomous_direction_field']


# Autonomous systems of ODEs:

def ode_system_from_matrix(mat):
    """Generates a system of ODEs from a given matrix.

    Parameters:
        mat: An `n` by `n` list/array which represents the system.
            Note that only `2x2` systems can be visualized.

    Returns:
        A method which can be called with `n` inputs that returns
        the values as dictated by the input system.
    """
    mat = np.array(mat)

    # Generate the application method.
    def apply(*args):
        return [sum([mat[i][j] * value for j, value in zip(
            (_ for _ in range(mat.shape[1])), args)])
                for i in range(mat.shape[0])]

    return apply


def plot_autonomous_direction_field(dv, x_range, y_range, grid_vecs=15,
                                    *, ic=None, cp=None, t_range=(-10, 10),
                                    figsize=None, grid=True,
                                    normalize_quiver=False, quiver_params=None):
    """Plots an autonomous direction field of vectors.

    Parameters:
        dv: A function which accepts two arguments, `x` and `y`,
            and returns the system of derivatives of these two
            inputs (should be a numpy array of two values).
        x_range: A two-tuple with the min/max values for the x-axis.
        y_range: A two-tuple with the min/max values for the y-axis.
        grid_vecs: An optional tuple with the number of vectors
            to plot along the x- and y-axes. Defaults to (15, 15).
        ic: An optional tuple of initial conditions for the equation.
            This can also be up to 5 distinct initial conditions.
        cp: An optional tuple of critical point coordinates for the
            system. This can optionally be a list of an arbitrary
            number of critical points. Note that if passing a `system`
            object, then this method will automatically extract
            the critical points from the class property.
        t_range: An optional range of `t` values for initial values.
            Defaults to -10 to 10 if not provided and ICs are.
        figsize: An optional tuple containing the size of the figure.
        grid: Whether to keep the grid on or off.
        normalize_quiver: Whether to normalize the length of the
            quiver arrows, e.g. set them to a constant length.
        quiver_params: An optional dictionary with parameters for the
            `plt.quiver` method which plots the vectors.
    """
    # Unpack the axis ranges and create the vector grid.
    x_min, x_max = x_range
    y_min, y_max = y_range
    if isinstance(grid_vecs, int):
        grid_vecs = (grid_vecs, grid_vecs)
    x = np.linspace(x_min, x_max, grid_vecs[0])
    y = np.linspace(y_min, y_max, grid_vecs[1])
    X, Y = np.meshgrid(x, y)

    # Calculate the derivatives from the system.
    dX, dY = dv(X, Y)

    # Create the figure.
    if quiver_params is None:
        quiver_params = {}
    quiver_params.setdefault('scale', 15)
    quiver_params.setdefault('headwidth', 5)
    if figsize is None:
        figsize = (6, 6)
    plt.figure(figsize=figsize)

    # Normalize the arrows if requested to.
    if normalize_quiver:
        dX /= np.sqrt(dX ** 2 + dY ** 2) + 1e-6
        dY /= np.sqrt(dX ** 2 + dY ** 2) + 1e-6

    # Plot the direction field.
    plt.quiver(X, Y, dX, dY, pivot='mid', color='k', zorder=5, **quiver_params)
    plt.xlim(x_range), plt.ylim(y_range)
    if grid:
        plt.grid('on', zorder=0)

    # Plot the critical points.
    if cp is not False:
        if hasattr(cp, 'critical_points'):
            cp = dv.critical_points
        if cp is not None:
            cp = np.array(cp)
            plt.scatter(cp[:, 0], cp[:, 1], c='b', s=10,
                        edgecolors='b', alpha=0.8, linewidths=5, zorder=10)

    # If initial conditions are provided, plot a trajectory.
    if ic is not None:
        colors = ['red', 'green', 'yellow', 'orange', 'purple']
        if isinstance(ic[0], int):
            ic = [ic]
        assert len(ic) <= len(colors), \
            f"Cannot have more than {len(colors)} initial conditions."

        for idx, cond in enumerate(ic):
            # Modify the signature of `dv` for the solver.
            def dv_mod(s, t):
                return dv(s[0], s[1])

            cond = np.array([cond[0], cond[1]])
            t = np.linspace(t_range[0], t_range[1], (t_range[1] - t_range[0]) * 10)
            s = odeint(dv_mod, cond, t)
            plt.plot(s[:, 0], s[:, 1], color=colors[idx], zorder=15)

    # Display the figure.
    plt.show()

