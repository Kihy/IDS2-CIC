from pyswarms.backend.topology import Topology
import logging
import numpy as np
from pyswarms.utils.reporter import Reporter
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.backend import operators as ops
from pyswarms.backend.swarms import Swarm
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import cKDTree


def plot_contour(pos_history,canvas=None,title="Trajectory",mark=None,designer=None,mesher=None,animator=None,**kwargs):
    """Draw a 2D contour map for particle trajectories
    Here, the space is represented as a flat plane. The contours indicate the
    elevation with respect to the objective function. This works best with
    2-dimensional swarms with their fitness in z-space.
    Parameters
    ----------
    pos_history : numpy.ndarray or list
        Position history of the swarm with shape
        :code:`(iteration, n_particles, dimensions)`
    canvas : (:obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`),
        The (figure, axis) where all the events will be draw. If :code:`None`
        is supplied, then plot will be drawn to a fresh set of canvas.
    title : str, optional
        The title of the plotted graph. Default is `Trajectory`
    mark : tuple, optional
        Marks a particular point with a red crossmark. Useful for marking
        the optima.
    designer : :obj:`pyswarms.utils.formatters.Designer`, optional
        Designer class for custom attributes
    mesher : :obj:`pyswarms.utils.formatters.Mesher`, optional
        Mesher class for mesh plots
    animator : :obj:`pyswarms.utils.formatters.Animator`, optional
        Animator class for custom animation
    **kwargs : dict
        Keyword arguments that are passed as a keyword argument to
        :obj:`matplotlib.axes.Axes` plotting function
    Returns
    -------
    :obj:`matplotlib.animation.FuncAnimation`
        The drawn animation that can be saved to mp4 or other
        third-party tools
    """

    try:
        # If no Designer class supplied, use defaults
        if designer is None:
            designer = Designer(
                limits=[(-1, 1), (-1, 1)], label=["x-axis", "y-axis"]
            )

        # If no Animator class supplied, use defaults
        if animator is None:
            animator = Animator()

        # If ax is default, then create new plot. Set-up the figure, the
        # axis, and the plot element that we want to animate
        if canvas is None:
            fig, ax = plt.subplots(1, 1, figsize=designer.figsize)
        else:
            fig, ax = canvas

        # Get number of iterations
        n_iters = len(pos_history)

        # Customize plot
        ax.set_title(title, fontsize=designer.title_fontsize)
        ax.set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
        ax.set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
        ax.set_xlim(designer.limits[0])
        ax.set_ylim(designer.limits[1])

        # Make a contour map if possible
        if mesher is not None:
            xx, yy, zz, = _mesh(mesher)
            ax.contour(xx, yy, zz, levels=mesher.levels)

        # Mark global best if possible
        if mark is not None:
            ax.scatter(mark[0], mark[1], color="red", marker="x")

        n_particles=pos_history[0].shape[0]

        # last particle is global best
        colours=["black"  for i in range(n_particles//2)]+["red" for i in range(n_particles//2)]

        # Put scatter skeleton
        plot = ax.scatter(x=[0 for i in range(n_particles)], y=[0 for i in range(n_particles)], c=colours, alpha=0.6, **kwargs)
        # Do animation
        anim = animation.FuncAnimation(
            fig=fig,
            func=_animate,
            frames=range(n_iters),
            fargs=(pos_history, plot),
            interval=animator.interval,
            repeat=animator.repeat,
            repeat_delay=animator.repeat_delay,
        )
    except TypeError:
        rep.logger.exception("Please check your input type")
        raise
    else:
        return anim

def _animate(i, data, plot):
    """Helper animation function that is called sequentially
    :class:`matplotlib.animation.FuncAnimation`
    """
    current_pos = data[i]
    if np.array(current_pos).shape[1] == 2:
        plot.set_offsets(current_pos)
    else:
        plot._offsets3d = current_pos.T
    return (plot,)

def decay_factor(i):
    return np.exp(-0.1*i)

def boost_factor(i):
    # cap factor at 20 iterations
    i=np.where(i>10,10,i)
    i=np.expand_dims(i, axis=1)
    return np.exp(0.05*i)

def create_swarm(n_particles, dimensions, options, bounds, discrete_index):

    position=generate_position(n_particles, dimensions, bounds, discrete_index)
    velocity=generate_velocity(n_particles, dimensions,bounds, discrete_index)

    swarm= Swarm(position, velocity, options=options)
    swarm.discrete_index=discrete_index
    return swarm

def generate_velocity(n_particles, dimensions,bounds,discrete_index):
    lb, ub = bounds
    range=ub-lb
    factor=0.2

    #set velocity to random value between 0 to factor of range
    v_cont=np.random.uniform(0, factor*range[:discrete_index], size=(n_particles, discrete_index))

    v_disc=np.random.randint(0, np.rint(factor*range[discrete_index:]),size=(n_particles, dimensions-discrete_index))

    return np.hstack((v_cont, v_disc))

def generate_position(n_particles, dimensions, bounds, discrete_index):
    # assume discrete dimensions are all after continous ones
    lb, ub = bounds

    # assume discrete dimensions are all after continous ones
    pos_cont = np.random.uniform(
        low=lb[:discrete_index], high=ub[:discrete_index], size=(n_particles, discrete_index)
    )
    pos_disc=np.random.randint(low=lb[discrete_index:], high=ub[discrete_index:],size=(n_particles, dimensions-discrete_index))

    return np.hstack((pos_cont,pos_disc))

class Traffic(Topology):
    def __init__(self,static=True):
        super(Traffic, self).__init__(static)
        self.rep=Reporter(logger=logging.getLogger(__name__))

        # contains information about randomly generated packets
        self.auxiliary_info=[]

    def compute_gbest_local(self, swarm, p, k, **kwargs):
        try:
            # Check if the topology is static or not and assign neighbors
            if (self.static and self.neighbor_idx is None) or not self.static:
                # Obtain the nearest-neighbors for each particle
                tree = cKDTree(swarm.position)
                _, self.neighbor_idx = tree.query(swarm.position, p=p, k=k)

            # Map the computed costs to the neighbour indices and take the
            # argmin. If k-neighbors is equal to 1, then the swarm acts
            # independently of each other.
            if k == 1:
                # The minimum index is itself, no mapping needed.
                self.neighbor_idx = self.neighbor_idx[:, np.newaxis]
                best_neighbor = np.arange(swarm.n_particles)
            else:
                idx_min = swarm.pbest_cost[self.neighbor_idx].argmin(axis=1)
                best_neighbor = self.neighbor_idx[
                    np.arange(len(self.neighbor_idx)), idx_min
                ]

            # Obtain best cost and position
            best_cost = np.min(swarm.pbest_cost[best_neighbor])
            best_pos = swarm.pbest_pos[best_neighbor]
            best_aux=swarm.pbest_aux[best_neighbor]
        except AttributeError:
            self.rep.logger.exception(
                "Please pass a Swarm class. You passed {}".format(type(swarm))
            )
            raise
        else:
            return best_pos, best_cost, best_aux

    def compute_gbest(self, swarm, **kwargs):
        if np.min(swarm.pbest_cost) < swarm.best_cost:
            # Get the particle position with the lowest pbest_cost
            # and assign it to be the best_pos
            min_index=np.argmin(swarm.pbest_cost)
            best_pos = swarm.pbest_pos[min_index]
            best_cost = swarm.pbest_cost[min_index]
            best_aux=swarm.pbest_aux[min_index]
        else:
            # Just get the previous best_pos and best_cost
            best_pos, best_cost,best_aux = swarm.best_pos, swarm.best_cost,swarm.best_aux

        # print(swarm.pbest_cost)
        # print(swarm.pbest_pos)
        # print(swarm.pbest_aux)
        # print(min_index)


        return best_pos, best_cost,best_aux

    def compute_pbest(self, swarm, iter):
         # Infer dimensions from positions
        dimensions = swarm.dimensions
        # Create a 1-D and 2-D mask based from comparisons
        mask_cost = swarm.current_cost < swarm.pbest_cost
        mask_pos = np.expand_dims(mask_cost ,axis=1)
        # Apply masks
        new_pbest_pos = np.where(~mask_pos, swarm.pbest_pos, swarm.position)
        new_pbest_cost = np.where(
            ~mask_cost, swarm.pbest_cost, swarm.current_cost
        )
        # print("pos",swarm.pbest_pos.shape,swarm.position.shape)
        # print("aux",swarm.pbest_aux.shape,swarm.current_aux.shape)
        # print("mask", mask_cost.shape)
        # apply aux info
        new_pbest_aux = np.where(
            ~mask_pos, swarm.pbest_aux, swarm.current_aux
        )

        # record the iteration with best cost
        new_pbest_iter=np.where(
            ~mask_cost, swarm.pbest_iter, iter
        )
        return (new_pbest_pos, new_pbest_cost, new_pbest_aux, new_pbest_iter)


    def compute_velocity(
        self,
        swarm,
        clamp=None,
        vh=VelocityHandler(strategy="unmodified"),
        bounds=None,
        iter=None
    ):
        """Compute the velocity matrix
        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm.
        A sample usage can be seen with the following:
        .. code-block :: python
            import pyswarms.backend as P
            from pyswarms.backend.swarm import Swarm
            from pyswarms.backend.handlers import VelocityHandler
            from pyswarms.backend.topology import Star
            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Star()
            my_vh = VelocityHandler(strategy="adjust")
            for i in range(iters):
                # Inside the for-loop
                my_swarm.velocity = my_topology.update_velocity(my_swarm, clamp, my_vh,
                bounds)
        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        clamp : tuple of floats (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh : pyswarms.backend.handlers.VelocityHandler
            a VelocityHandler instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        Returns
        -------
        numpy.ndarray
            Updated velocity matrix
        """
        swarm_size = swarm.position.shape
        c1 = swarm.options["c1"]
        c2 = swarm.options["c2"]
        w = swarm.options["w"]
        # Compute for cognitive and social terms
        cognitive = (
            c1
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.pbest_pos - swarm.position)
        )
        social = (
            c2
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.best_pos - swarm.position)
        )
        # Compute temp velocity (subject to clamping if possible)
        temp_velocity = (w * swarm.velocity) + cognitive + social
        temp_velocity *= boost_factor(iter-swarm.pbest_iter)

        updated_velocity = vh(
            temp_velocity, clamp, position=swarm.position, bounds=bounds
        )


        updated_velocity[:,swarm.discrete_index:]=np.rint(updated_velocity[:,swarm.discrete_index:])

        return updated_velocity

    def compute_position(
        self, swarm, bounds=None, bh=BoundaryHandler(strategy="random")
    ):
        """Update the position matrix
        This method updates the position matrix given the current position and
        the velocity. If bounded, it waives updating the position.
        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh : pyswarms.backend.handlers.BoundaryHandler
            a BoundaryHandler instance
        Returns
        -------
        numpy.ndarray
            New position-matrix
        """

        temp_position = swarm.position.copy()
        temp_position += swarm.velocity

        if bounds is not None:
            temp_position = bh(temp_position, bounds)

        temp_position[:,swarm.discrete_index:]=np.rint(temp_position[:,swarm.discrete_index:])

        position = temp_position

        return position
