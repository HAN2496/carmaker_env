from array import array
from time import sleep, time
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import figaspect
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy import sin, cos, hstack, min as npmin, max as npmax, asarray, zeros, zeros_like
from numpy.linalg import norm

class LivePlot:

    def __init__(self, track, vehicle_model, mode="human", vmin=0.0, vmax=14, num_trajectories=2):
        
        self.mode = mode
        self.track = track
        self.model = vehicle_model
        self.vmin = vmin
        self.vmax = vmax

        xmin =  0.0
        xmax =  400.0
        ymin = -5.0
        ymax =  5.0

        margin = max(xmax - xmin, ymax - ymin) * 0.01
        
        self.figsize = figaspect((ymax - ymin + 2*margin) / (xmax - xmin + 2*margin))

        if self.mode!="raw_data":
            self.fig = plt.figure(figsize=self.figsize)
            self.ax = plt.gca()
            self.ax.set_xlim(
                left  = xmin - margin,
                right = xmax + margin
            )
            self.ax.set_ylim(
                bottom = ymin - margin,
                top    = ymax + margin
            )
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('x[m]')
            self.ax.set_ylabel('y[m]')

            self.track_artists = self.get_track_artists(self.ax)

            self.vehicle_artists = self.get_vehicle_artists(self.ax, num_trajectories=num_trajectories)

            if self.mode=="human":
                self.fig.canvas.draw()
            if self.mode in {"rgb_array", "single_rgb_array"}:
                self.canvas_agg = FigureCanvasAgg(self.fig)
                self.canvas_agg.draw()
            for a in self.track_artists:
                self.ax.draw_artist(a)
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)


    def show(self):
        self.fig.show()


    def start_event_loop(self, timeout=0.001):
        self.fig.canvas.start_event_loop(timeout)


    def update(self, state, trajectories=None, collided_cone=None):
        if self.mode=="raw_data":
            self.state = state
            if trajectories is None:
                self.trajectories = [zeros((0, 8)),  zeros((0, 2))]
            else:
                self.trajectories = trajectories
        else:
            self.fig.canvas.restore_region(self.background)
            if trajectories is None:
                trajectories = [None, None]
            for i, trajectory in enumerate(trajectories):
                self.update_vehicle_artists(self.vehicle_artists[i], state, trajectory)
                for a in self.vehicle_artists[i]:
                    self.ax.draw_artist(a)
            if self.mode=="human":
                self.fig.canvas.blit(self.ax.bbox)
    

    def get_data(self):
        if self.mode in {"rgb_array", "single_rgb_array"}:
            return asarray(self.canvas_agg.buffer_rgba()).copy()
        elif self.mode=="raw_data":
            return self.state.copy(), self.trajectories.copy()
        else:
            return None
    

    def get_track_artists(self, ax):
        return [
            ax.plot([0.0, 400.0], [ 0.0,  0.0], '--k', linewidth=0.5, animated=True)[0],
            ax.plot([0.0, 400.0], [ 5.0,  5.0], color='k', linewidth=1, animated=True)[0],
            ax.plot([0.0, 400.0], [-5.0, -5.0], color='k', linewidth=1, animated=True)[0],
            ax.scatter(
                self.track.cone_position, zeros_like(self.track.cone_position), c="r",
                s=20, vmin=self.vmin, vmax=self.vmax,
                edgecolor='none', marker='o', animated=True
            )
        ]


    def get_vehicle_artists(self, ax, num_trajectories):
        return [
            [
                ax.scatter(
                    [], [], c=[],
                    s=16, vmin=self.vmin, vmax=self.vmax,
                    cmap=cm.rainbow, edgecolor='none', marker='o', animated=True
                ),
                ax.plot([], [], color='k', linewidth=1, animated=True)[0]
            ] for _ in range(num_trajectories)
        ]


    def update_vehicle_artists(self, vehicle_artists, state, trajectory):
        if trajectory is not None:
            vehicle_artists[0].set_offsets(
                trajectory[:, :2] # X, Y
            )
            if trajectory.shape[1] > 2:
                vehicle_artists[0].set_array(
                    norm(trajectory[:, 3:5], axis=1) # norm of vx, vy
                )
            else:
                vehicle_artists[0].set_array(
                    zeros((trajectory.shape[0], ))
                )
        else:
            vehicle_artists[0].set_offsets(zeros((0, 2)))
            vehicle_artists[0].set_array(zeros((0,)))
        vx, vy = self.model.footprint(state[0], state[1], state[2])
        vehicle_artists[1].set_data(
            hstack((vx, vx[:1])), hstack((vy, vy[:1]))
        )
        return vehicle_artists


    def reset_vehicle_artists(self, vehicle_artists):
        for artists in vehicle_artists:
            #vehicle_artists[0].set_offsets(
            #    [[]]
            #)
            vehicle_artists[0].set_offsets(
                zeros((0, 2))
            )
            vehicle_artists[0].set_array(
                []
            )
            vehicle_artists[1].set_data(
                [], []
            )
        return vehicle_artists



class Clock:

    def __init__(self):
        self.time_last_tick = None
        self.duration = None

    def tick(self, framerate):
        if self.time_last_tick is None:
            self.time_last_tick = time()
        else:
            time_current_tick = time()
            timeout = self.time_last_tick + 1.0/framerate - time_current_tick
            if timeout > 0.0:
                sleep(timeout)
                time_current_tick = time()
            self.duration = time_current_tick - self.time_last_tick
            self.time_last_tick = time_current_tick

    def get_fps(self):
        if self.duration is None:
            return -1.0
        else:
            return 1.0 / self.duration
