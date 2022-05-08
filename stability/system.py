"""
Systems of Autonomous Ordinary Differential Equations

Used for analyzing direction fields of systems of autonomous ODEs,
determining trajectories of initial conditions, and finding stability,
point classification, and locally linear systems of critical points.
"""

import os
import re
import warnings
from tqdm import tqdm

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from sympy.abc import *
from sympy import solve, Derivative, Matrix, nsimplify


class StabilityAnalysis(object):
    """Wrapper class which displays an analysis of critical point stabilities."""
    def __init__(self, eigenvalues, jacobians):
        self._cp_types = {}
        self._stabilities = {}
        self._original_mapping = eigenvalues.copy()
        self._jacobians = jacobians.copy()
        for cp, eigenvalue in self._original_mapping.items():
            cp_type, stability = self._determine_stability(eigenvalue)
            self._cp_types[cp] = cp_type
            self._stabilities[cp] = stability

    def __repr__(self):
        ret = []
        fmt = "{0:<19}{1:<26}{2:<24}{3}"
        lsys_format = "d/dt(x, y) = ({0}"
        ret.append(fmt.format(
            'Critical Point', 'Type of Critical Point',
            'Stability', 'Linear System') + '\n')
        ret.append('-' * (len(ret[0]) + 18) + '\n')
        for cp in self._stabilities.keys():
            p_cp = f"({nsimplify(f'{cp[0]}')}, {nsimplify(f'{cp[1]}')})"
            if len(p_cp) > 19:
                p_cp = f"({round(cp[0], 3)}, {round(cp[1], 3)})"
            out = fmt.format(
                p_cp, self._cp_types[cp],
                self._stabilities[cp],
                lsys_format.format(
                    [nsimplify(i) for i in self._jacobians[cp].tolist()[0]]))
            out += "\n" + " " * out.index('[') + \
                   f"{[nsimplify(i) for i in self._jacobians[cp].tolist()[1]]})(x, y)\n"
            ret.append(out)
        return ''.join(ret)

    def __getitem__(self, item):
        if item not in self._stabilities.keys():
            raise KeyError("Expected one of the critical points.")
        return {'Critical Point Type': self._cp_types[item],
                'Stability': self._stabilities[item]}

    @property
    def info(self):
        return {i: self[i] for i in self._stabilities.keys()}

    @staticmethod
    def _is_real(value):
        return isinstance(value, float)

    @staticmethod
    def _is_complex(value):
        return isinstance(value, complex)

    def _determine_stability(self, eigenvalues):
        e1, e2 = eigenvalues
        if self._is_real(e1) and self._is_real(e2):
            if e1 == e2:  # repeated real eigenvalues
                node = 'Proper or Improper Node'
                if e1 > 0:
                    stability = 'Unstable'
                else:
                    stability = 'Asymptotically Stable'
                return node, stability
            else:  # two distinct real eigenvalues
                if e1 > e2 >= 0 or e2 > e1 >= 0:
                    return 'Node', 'Unstable'
                if e1 < e2 <= 0 or e2 < e1 <= 0:
                    return 'Node',  'Asymptotically Stable'
                if e1 <= 0 <= e2 or e2 <= 0 <= e1:
                    return 'Saddle Point', 'Unstable'
        else:  # complex eigenvalues
            if not self._is_complex(e1) and self._is_complex(e2):
                raise ValueError(
                    f"Error in computing stability: got one real "
                    f"and one complex eigenvalue: {e1, e2}.")
            real_part = e1.real
            if real_part > 0:
                return 'Spiral Point', 'Unstable'
            elif real_part < 0:
                return 'Spiral Point', 'Asymptotically Stable'
            else:
                return 'Center', 'Stable'


class system(object):
    """Generates a system of differential equations.

    You can instantiate a `system` object using a string expression:

    > sys = system('2x - 3, xy^2')

    This assumes that you are using a system dependent on two arbitrary
    variables (traditionally this would be `x` and `y`, but any two
    letters are supported as long as they are the only two letters used).

    The following conditions apply for typing:

    1. Any numbers which happen to precede these two variables will be
       considered coefficients of the expression (e.g., `2x` would be
       expanded to `2 * x`, and `245xy` would be `245 * x * y`).
    2. Exponents can be achieved using either the `**` or `^` symbol.
       If you want an entire expression to be squared, then either wrap
       it in parenthesis, brackets, or braces (e.g., `2x ** 2` would
       be expanded to `2 * x ** 2`, while `2 ** 2x` would be expanded
       to `2 ** (2 * x)`, and `(2x) ** 2` would be `(2 * x) ** 2`).
    3. Different terms are distinguished by use of spaces or operators
       such as `+` or `-` (e.g., `x-y` would be `x - y`). If there are
       two digits next to each other they will be deemed as one single
       number (e.g., `21xy` will be `21 * x * y`).

    You can either pass two expressions or use a comma in a single string to
    indicate where the two systems are independent.

    Parameters:
        sys: The first equation, or a string containing the whole system.
        sys2: The second equation if passing independent strings.
    """
    # Stores predefined SymPy symbols for variables.
    _symbol_cache = {}

    # Allowed letter phrases for trigonometry.
    _allowed_phrases = ('sin', 'cos', 'sec', 'csc', 'tan', 'cot')
    _allowed_phrases = _allowed_phrases + tuple(
        ['arc' + i for i in _allowed_phrases]) + ('e', 'sqrt')

    @classmethod
    def matrix(cls, mat):
        """Creates a system from a 2x2 matrix of coefficients.

        If you want to make a system which follows the equation format
        of x' = Ax, then this does that from the matrix `A`.

        Parameters:
            mat: A 2x2 matrix (numpy array, list of lists), or a
                 single 1-dimensional list with 4 values.
        """
        mat = np.array(mat)
        if mat.ndim == 1:
            mat = mat.reshape((2, 2))
        return cls(
            f'{mat[0][0]}x + {mat[0][1]}y',
            f'{mat[1][0]}x + {mat[1][1]}y')

    @classmethod
    def polar(cls, sys, sys2):
        """Creates a system from a set of polar equations.

        Polar equations should use the variables `r` and `theta`.

        Parameters:
            sys: A polar equation.
            sys2: A second polar equation.
        """
        sys = sys.replace('r', 'x')
        sys = sys.replace('theta', 'y')
        sys2 = sys2.replace('r', 'x')
        sys2 = sys2.replace('theta', 'y')
        return cls(sys, sys2, polar=True)

    def __init__(self, sys, sys2=None, **kwargs):
        # Get the two independent systems.
        if sys2 is None:
            if ',' not in sys:
                raise ValueError(
                    "Got only one string, but it contains no comma. "
                    "Expected two equations for a system.")
            expr1, expr2 = sys.split(',')
        else:
            expr1, expr2 = sys, sys2

        # Pre-format the expressions for potential letters/phrases.
        expr1 = expr1.strip()
        expr2 = expr2.strip()
        expr1 = expr1.replace(' ', '')
        expr2 = expr2.replace(' ', '')
        expr1, fmt_1 = self._pre_alpha_format(expr1)
        expr2, fmt_2 = self._pre_alpha_format(expr2)

        # Parse and format the expressions, and get the
        # variables which are being used by the system.
        expr1, var1 = self._parse_expr(expr1)
        expr2, var2 = self._parse_expr(expr2)

        # Check the variables.
        all_vars = np.unique(var1 + var2)
        if len(all_vars) > 2:
            raise ValueError(
                f"Got more than 2 independent variables: {all_vars}.")
        self._vars = all_vars

        # Post-format the expression.
        expr1 = self._post_alpha_format(expr1, fmt_1)
        expr2 = self._post_alpha_format(expr2, fmt_2)

        # Set the printing version.
        self._pprint_exprs = (expr1, expr2)

        # Expand the coefficients to proper expressions and compile them
        # for evaluation, but store the originals as a property.
        self.expr1 = self._expand_coefficients(expr1, all_vars)
        self.expr2 = self._expand_coefficients(expr2, all_vars)
        self._expressions = (self.expr1, self.expr2)

        # Parse for a polar equation.
        self._polar = kwargs.get('polar', False)
        if self._polar:
            self._vars = np.array(['r', 'theta'])
            self.expr1 = self.expr1.replace('x', 'r')
            self.expr1 = self.expr1.replace('y', 'theta')
            self.expr2 = self.expr2.replace('x', 'r')
            self.expr2 = self.expr2.replace('y', 'theta')
            self._expressions = (self.expr1, self.expr2)

        # Find the critical points of the expression.
        cp = kwargs.get('cp', True)
        self._cps = None
        if cp:
            sympy_vars = self._gen_sympy_symbols(self._vars)
            try:
                cps = solve(self._expressions, *sympy_vars, set=True)[1]
            except IndexError:
                cps = None
            self._cps = self._parse_nums(cps)

        # Parse the expressions again for evaluation (`sin` -> `np.sin`).
        self.expr1 = self._parse_for_eval(self.expr1)
        self.expr2 = self._parse_for_eval(self.expr2)

    def __repr__(self):
        if self._polar:
            dx, dy = 'dr/dt', 'd0/dt'
        else:
            dx, dy = 'dx/dt', 'dy/dt'
        return ("system("
                f"{dx} = {self._repr_expr(self._pprint_exprs[0])}"
                ", "
                f"{dy} = {self._repr_expr(self._pprint_exprs[1])}"
                ")")

    @property
    def expressions(self):
        return self._expressions

    @property
    def critical_points(self):
        return self._cps

    @property
    def real_critical_points(self):
        if self._cps is None:
            return None
        return [i for i in self._cps if not isinstance(i[0], complex)]

    def __call__(self, x, y):
        if self._polar:
            r, theta = (x ** 2 + y ** 2) ** (1/2), np.arctan2(y, x)
            ret1 = eval(self.expr1, globals(), locals())
            ret2 = eval(self.expr2, globals(), locals())
            res1 = (r + ret1) * np.cos(theta + ret2) - x
            res2 = (r + ret1) * np.cos(theta + ret2) - y
        else:
            res1 = eval(self.expr1, globals(), locals())
            res2 = eval(self.expr2, globals(), locals())
        return np.array([res1, res2])

    @staticmethod
    def _repr_expr(expr):
        expr = expr.replace('**', '~~')  # for parsing multiplication.
        symbol_parse = lambda x, expr: f" {x} ".join(expr.split(f'{x}'))
        for sym in ['+', '-', '**']:
            expr = symbol_parse(sym, expr)
        expr = expr.replace('~~', '**')
        expr = symbol_parse('**', expr)
        expr = expr.replace('  ', ' ').replace('  ', ' ')
        expr = expr.replace('( -', '(-')
        return expr

    def _pre_alpha_format(self, expr):
        fmt_list = []
        for phrase in self.__class__._allowed_phrases:
            if expr.find(phrase) != -1:
                fmt_list.append(phrase)
            expr = expr.replace(phrase, '~')
        return expr, fmt_list

    @staticmethod
    def _post_alpha_format(expr, fmt_dict):
        manip_offset = 0
        while True:
            occur = [m.start(0) for m in re.finditer('~', expr)]
            if len(occur) == 0:
                break
            expr = expr[:occur[0]] + fmt_dict[manip_offset] + \
                   expr[occur[0] + 1:] # noqa
            manip_offset += 1
        return expr

    @staticmethod
    def _parse_expr(expr):
        # Replace any `^` exponential with `**`.
        expr = expr.replace('^', '**')

        # Get the two independent variables.
        letters = "".join(re.findall("[a-zA-Z]+", expr))

        # Replace all brackets and braces with parenthesis.
        expr = expr.replace(r'\{', '(')
        expr = expr.replace('[', '(')
        expr = expr.replace(r'\}', ')')
        expr = expr.replace(']', ')')

        # Return the system + variables.
        return expr, list(iter(letters))

    @staticmethod
    def _expand_coefficients(expr, var):
        def _regen_parser(expr_):
            return np.array(list(iter(expr_)))

        # Get the locations of the variables and find any coefficients before
        # variables (also track the expansion of the string since it is being
        # manipulated while we are parsing through the loop).
        for v in [*var, *['s', 'c', 't']]:  # trigonometry
            manip_tracker = 0
            parser = _regen_parser(expr)
            loc = np.where(parser == v)[0]
            for idx in loc:
                if idx != 0:
                    if parser[idx - 1].item().isdigit():
                        num_indexes, start_pos = [idx - 1], 2
                        try:
                            while True:
                                if idx - start_pos != 0:
                                    break
                                if parser[idx - start_pos].item().isdigit():
                                    num_indexes.append(idx - start_pos)
                                    start_pos += 1
                                else:
                                    break
                        except IndexError:
                            pass

                        # Verify that it is a valid number and expand.
                        if ''.join(parser[num_indexes]).isdigit():
                            insert_index = max(num_indexes) + 1 + manip_tracker
                            expr = expr[:insert_index] + '*' + expr[insert_index:]
                            manip_tracker += 1

                    if parser[idx - 1].item() in var:
                        insert_index = idx + manip_tracker
                        expr = expr[:insert_index] + '*' + expr[insert_index:]
                        manip_tracker += 1

        # For any sets of multiplied parenthesis, add a `*`.
        manip_tracker = 0
        parser = _regen_parser(expr)
        paren_loc = np.where(parser == ')')[0]
        for loc in paren_loc:
            if loc + 1 + manip_tracker == len(expr):
                continue
            if (expr[loc + manip_tracker + 1] == '('
                    or expr[loc + manip_tracker + 1].isalpha()
                    or expr[loc + manip_tracker + 1].isdigit()):
                if (expr[loc + manip_tracker - 1].isalpha() and
                        expr[loc + manip_tracker - 1] in ['n', 's', 'c', 't']):
                    continue
                insert_index = loc + 1 + manip_tracker
                expr = expr[:insert_index] + '*' + expr[insert_index:]
                manip_tracker += 1

        # Do the same multiplication for the reverse case, namely
        # when there is a variable or other expression before a
        # parenthesis, like `2(x + y)` or `x(2 + x)`.
        manip_tracker = 0
        parser = _regen_parser(expr)
        paren_loc = np.where(parser == '(')[0]
        for loc in paren_loc:
            if loc + 1 + manip_tracker == len(expr):
                continue
            if (expr[loc + manip_tracker - 1] == '('
                    or expr[loc + manip_tracker - 1].isalpha()
                    or expr[loc + manip_tracker - 1].isdigit()):
                if (expr[loc + manip_tracker - 1].isalpha() and
                        expr[loc + manip_tracker - 1] in ['n', 's', 'c', 't']):
                    continue
                insert_index = loc + manip_tracker
                expr = expr[:insert_index] + '*' + expr[insert_index:]
                manip_tracker += 1

        # Return the parsed string expression.
        return expr

    def _parse_for_eval(self, expr):
        phrase_index = 0
        while True:
            try:
                phrase = self.__class__._allowed_phrases[phrase_index]
            except IndexError:
                break
            if expr.find(phrase) != -1:
                sub = 'np.' + phrase
                expr = expr[:expr.find(phrase)] + sub + \
                       expr[expr.find(phrase) + len(phrase):] # noqa
            phrase_index += 1
        return expr

    @classmethod
    def _gen_sympy_symbols(cls, sym):
        ret_symbols, exist_symbols = [0 for _ in sym], []
        for idx, var in enumerate(sym):
            if var in cls._symbol_cache.keys():
                ret_symbols[idx] = cls._symbol_cache[var]
                exist_symbols.append(ret_symbols[idx])
        non_symbols = np.where(np.array(ret_symbols) == 0)[0]
        new_symbols = np.array(sym)[non_symbols]
        if len(new_symbols) != 0:
            new_gen_symbols = symbols(' '.join(new_symbols))
            if not isinstance(new_gen_symbols, (list, tuple)):
                new_gen_symbols = (new_gen_symbols, )
            for name, symbol in zip(new_symbols, new_gen_symbols):
                cls._symbol_cache[name] = symbol
            return (*exist_symbols, ) + (*new_gen_symbols, )
        return tuple(exist_symbols)

    @staticmethod
    def _parse_nums(nums):
        if nums is None:
            return nums
        nums = tuple(nums)
        if len(nums) == 0:
            return None
        if not hasattr(nums[0], '__len__'):
            try:
                return [(float(nums[0]), float(nums[1]))]
            except TypeError:
                # complex numbers (or critical points).
                cp_complex = ([complex(i) for i in nums])
                return [tuple([
                    c.real if c.imag == 0 else c for c in cp_complex])]
        try:
            return [
                tuple([float(i) for i in cp]) for cp in nums]
        except TypeError:
            try:
                # complex numbers (or critical points).
                cp_complex = [tuple([complex(i) for i in cp]) for cp in nums]
                return [
                    tuple([c.real if c.imag == 0 else c for c in cp])
                    for cp in cp_complex]
            except TypeError:
                return None

    def plot(self, x_range=None, y_range=None, grid_vecs=15,
             *, ic=None, cp=True, ba=False, t_range=(-50, 50),
             figsize=None, grid=True, plot_type='quiver',
             normalize_quiver=False, quiver_params=None,
             return_figure=False, save_figure=False) -> None:
        """Plots an autonomous direction field of vectors.

        Parameters:
            x_range: A two-tuple with the min/max values for the x-axis.
                This is optional, if unprovided, then `cp` must be set
                to `True` and the plot will be generated based on the
                critical points. If there are no critical points or the
                only critical point is `(0, 0)`, then the range defaults
                to `(-5, 5)` (for both axes).
            y_range: A two-tuple with the min/max values for the y-axis.
                This is optional, see `x_range` for more information. If
                you want the same range for both the `x` and `y` axes,
                however, then pass it to `x_range` and leave this as `None`.
            grid_vecs: An optional tuple with the number of vectors
                to plot along the x- and y-axes. Defaults to (15, 15).
            ic: An optional tuple of initial conditions for the equation.
                This can also be up to 5 distinct initial conditions.
            cp: Whether to display the critical points of the system. Is
                set to `True` by default, but can be toggled off.
            ba: An optional tuple of x and y coordinates for which the
                method will attempt to find a basin of attraction for.
            t_range: An optional range of `t` values for initial values.
                Defaults to -10 to 10 if not provided and ICs are.
            figsize: An optional tuple containing the size of the figure.
            grid: Whether to keep the grid on or off.
            plot_type: Whether to use `plt.streamplot` or `plt.quiver`,
                basically a stream field versus constant arrows for a
                direction field. Set to `quiver` by default, change to
                `stream` for a stream field. Set to `stream` by default
                for polar coordinates, however.
            normalize_quiver: Whether to normalize the length of the
                quiver arrows, e.g. set them to a constant length.
            quiver_params: An optional dictionary with parameters for the
                `plt.quiver` method which plots the vectors.
            return_figure: If set to `True`, this method will return the
                figure instead of displaying it (for saving, etc).
            save_figure: If you want to save the figure, then pass a file
                path to this argument and it will save it to that path.
        """
        # Determine the axis ranges.
        if x_range and not y_range:
            y_range = (x_range[0], x_range[1])
        if not all([i is None for i in [x_range, y_range]]):
            if not all([i is not None for i in [x_range, y_range]]):
                raise TypeError("Expected either both ranges or neither.")
        else:
            if self.real_critical_points is None:
                x_range, y_range = (-5, 5), (-5, 5)
            else:
                c_p = np.array(self.real_critical_points)
                if len(c_p) == 1 and (c_p == np.array([[0, 0]])).all():
                    x_range, y_range = (-5, 5), (-5, 5)
                else:
                    def _round_fourth(x):
                        return round(x * 4) / 4

                    xs, ys = c_p[:, 0], c_p[:, 1]
                    x_dist = xs.max() - xs.min()
                    if x_dist == 0.0:
                        x_dist = 5.0
                    x_range = (_round_fourth(xs.min() - 0.3 * x_dist),
                               _round_fourth(xs.max() + 0.3 * x_dist))
                    y_dist = ys.max() - ys.min()
                    if y_dist == 0.0:
                        y_dist = 5.0
                    y_range = (_round_fourth(ys.min() - 0.3 * y_dist),
                               _round_fourth(ys.max() + 0.3 * y_dist))

        # Unpack the axis ranges and create the vector grid.
        x_min, x_max = x_range
        y_min, y_max = y_range
        if isinstance(grid_vecs, int):
            grid_vecs = (grid_vecs, grid_vecs)
        x = np.linspace(x_min, x_max, grid_vecs[0])
        y = np.linspace(y_min, y_max, grid_vecs[1])
        X, Y = np.meshgrid(x, y)

        # Calculate the derivatives from the system.
        dX, dY = self(X, Y)

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
        if self._polar:
            plt.streamplot(X, Y, dX, dY, color='k', zorder=5)
        else:
            if plot_type == 'stream':
                plt.streamplot(X, Y, dX, dY, color='k', zorder=5)
            else:
                plt.quiver(X, Y, dX, dY, pivot='mid',
                           color='k', zorder=5, **quiver_params)
        plt.xlim(x_range), plt.ylim(y_range)
        if grid:
            plt.grid('on', zorder=0)

        # Plot the critical points.
        if not isinstance(cp, bool) or cp is True:
            if hasattr(cp, '__len__'):
                if not hasattr(cp[0], '__len__'): # noqa
                    cp = np.array([cp])
            else:
                if self.real_critical_points is not None:
                    cp = self.real_critical_points
                else:
                    cp = None
            if cp is not None:
                cp = np.array(self.real_critical_points)
                plt.scatter(cp[:, 0], cp[:, 1], c='b', s=10,
                            edgecolors='b', alpha=0.8, linewidths=5, zorder=10)

        # If initial conditions are provided, plot a trajectory.
        if ic is not None:
            colors = ['red', 'green', 'yellow', 'orange', 'purple']
            if isinstance(ic[0], (int, float)):
                ic = [ic]
            assert len(ic) <= len(colors), \
                f"Cannot have more than {len(colors)} initial conditions."

            for idx, cond in enumerate(ic):
                # Modify the signature of `dv` for the solver.
                def dv_mod(s, t):
                    return self(s[0], s[1])

                cond = np.array([cond[0], cond[1]])
                t = np.linspace(t_range[0], t_range[1], (t_range[1] - t_range[0]) * 10)
                s = odeint(dv_mod, cond, t)
                line = plt.plot(s[:, 0], s[:, 1], color=colors[idx], zorder=15)[0] # noqa

        # If a basin of attraction point is provided, try to find it.
        if ba:
            assert len(ba) == 2, "Expected one critical point for basin of attraction." # noqa
            ba = tuple([float(b) for b in ba]) # noqa
            assert ba in self.real_critical_points, "Expected a critical point."
            ba = list(ba)
            coords = list(zip(X.flat, Y.flat))

            # Find the trajectories for each coordinate, see if it falls in the basin.
            valid_trajectories, areas = {}, []
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                with tqdm(desc="Generating Basin of Attraction", leave=False,
                          total=len(coords)) as p_bar:
                    for idx, cond in enumerate(coords):
                        # Modify the signature of `dv` for the solver.
                        def dv_mod(s, t):
                            nonlocal self
                            return self(s[0], s[1])

                        cond = np.array([cond[0], cond[1]])
                        t = np.linspace(t_range[0], t_range[1],
                                        (t_range[1] - t_range[0]) * 10)
                        s = odeint(dv_mod, cond, t)
                        s = np.array([[round(x, 3) for x in i] for i in s])
                        s[s == -0.0] = 0.0
                        s = s.tolist()
                        if ba in s:
                            # Calculate the areas of the trajectories.
                            def _shoelace(co):
                                co = np.array(co)
                                x, y = co[:, 0], co[:, 1]
                                return 0.5 * np.abs(np.dot(x, np.roll(y, 1))
                                                    - np.dot(y, np.roll(x, 1)))
                            valid_trajectories[tuple(cond.tolist())] = s # noqa
                            areas.append(_shoelace(s))
                        p_bar.update(1)
                p_bar.close()

            # Filter through the trajectories for the outermost one.
            try:
                traj = valid_trajectories[list(valid_trajectories.keys())
                [np.argmax(areas)]] # noqa
            except ValueError:  # no basin
                pass
            else:
                traj = np.array(traj) # noqa
                plt.fill(traj[:, 0], traj[:, 1], color='blue')

        # Save the figure.
        if save_figure:
            savefig = plt.gcf()
            savefig.savefig(os.path.expanduser(save_figure))

        # Display the figure.
        if return_figure:
            return plt.gcf()
        plt.show()

    def analyze_stability(self, cp=None):
        """Analyzes the stability of the system's critical points.

        Parameters:
            cp: A specific critical point to analyze stability of.
        """
        # Construct an array representing the Jacobian:
        # [[∂F/∂x, ∂F/∂y], [∂G/∂x, ∂G/∂y]], although with
        # arbitrary variables substituted for `x` and `y`.
        symbols = self._gen_sympy_symbols(self._vars)
        dv = lambda expr, symbol: Derivative(expr, symbol).doit()
        expr1 = self.expr1.replace('np.', '')
        expr2 = self.expr2.replace('np.', '')
        jacobian = [dv(expr1, symbols[0]), dv(expr1, symbols[1]),
                    dv(expr2, symbols[0]), dv(expr2, symbols[1])]

        # Check if a specific critical point is provided. Otherwise,
        # determine the stability of all of the critical points.
        if cp is not None:
            if cp not in self.real_critical_points:
                if cp in self.critical_points:
                    raise NotImplementedError(
                        "Cannot analyze non-real critical points.")
                raise ValueError(f"Invalid critical point {cp}, "
                                 f"should be in {self.real_critical_points}")
            cp = [cp]
        else:
            cp = self.real_critical_points

        # Determine the eigenvalues for the points.
        eigenvalues, jacobians = {}, {}
        for point in cp:
            mat = []
            for expr in jacobian:
                expr = expr.subs(symbols[0], point[0])
                expr = expr.subs(symbols[1], point[1])
                mat.append(float(expr))
            mat = np.array(mat).reshape((2, 2))

            # Calculate eigenvalues.
            s_mat = Matrix(mat)
            values = s_mat.eigenvals(multiple=True)
            values = self._parse_nums(values)
            eigenvalues[point] = values[0]
            jacobians[point] = mat

        # Return the stability analysis.
        return StabilityAnalysis(eigenvalues, jacobians)

    def analyze(self, *args, **kwargs):
        """Run an analysis of the provided system.

        Namely, this method is a wrapped for the methods `system.plot` and
        `system.analyze_stability`, which respectively create a plot of
        the system and return an analysis of the stability of the eigenvalues.

        See those methods for a list of valid parameters.
        """
        cp = kwargs.pop('cp', None)
        self.plot(*args, **kwargs)
        return self.analyze_stability(cp)



