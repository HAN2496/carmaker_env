from typing import Optional
from time import asctime
from multiprocessing import Process

# from gym.core import Env
# from gym import spaces
from gymnasium import Env
from gymnasium import spaces

import numpy as np
from scipy.integrate import solve_ivp

from .renderer import Renderer
from .track import Track
from .live_plot import LivePlot, Clock
from .vehicle_model import VehicleModel
from .utils import RK4
from .config import Config



class SlalomTest(Env):

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array", "raw_data"],
        "render_fps": 60
    }

    def __init__(self, is_render=True, render_mode: Optional[str] = None):

        config = Config()
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.metadata["render_fps"] = config.env.render_fps
        self.is_X11_forwarding = config.env.is_X11_forwarding
        self.detach_video_export_process = config.env.detach_video_export_process
        self.video_export_location = config.env.video_export_location
        self.screen = None
        self.clock = None
        self.isopen = None
        self.is_render = is_render

        self.low = np.array([], dtype=np.float32)
        self.high = np.array([], dtype=np.float32)
        self.action_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        # self.dt = config.env.dt
        self.cone_radius = config.env.cone_radius
        self.Kv = config.env.Kv
        self.vref = config.env.vref
        self.model = VehicleModel(config)
        self.track = Track()
        self.lbu = np.array([config.vehicle.delta_min, config.vehicle.tau_min])
        self.ubu = np.array([config.vehicle.delta_max, config.vehicle.tau_max])
        self.lbdu = np.array([config.vehicle.ddelta_min, config.vehicle.dtau_min])
        self.ubdu = np.array([config.vehicle.ddelta_max, config.vehicle.dtau_max])

        self.integrator_type = config.env.integrator_type
        self.sim_method_num_steps = config.env.sim_method_num_steps

        self.state = None
        self.trajectory = None
        self.collided_cone = None

        self.late_start = config.env.late_start
        self.before_X = None


    def step(self, control, dt, trajectory=None):
        """
        control : Array of [ddelta, dtau].
        """
        if self.state is None:
            self.reset()
        self.trajectory = trajectory
        reward = 0.0
        terminated = bool(0.0)
        truncated = bool(0.0)

        sol = solve_ivp(
            lambda t, x: self.model.f_pp(
                *x[:-2],
                *np.clip(x[-2:],
                    [self.model.delta_min, self.model.tau_min], 
                    [self.model.delta_max, self.model.tau_max]
                ),
                *np.clip(control,
                    [self.model.ddelta_min, self.model.dtau_min], 
                    [self.model.ddelta_max, self.model.dtau_max]
                )
            ),
            (0, dt),
            self.state,
            method=self.integrator_type
        )
        self.state = sol.y[:, -1]

        if self.check_collision():
            terminated = bool(1.0)
        if self.state[0] > 400.0:
            truncated = bool(1.0)
        if self.state[3] < 1.5: # To prevent MPC from stopping.
            terminated = bool(1.0)       
        if self.state[1] >= 5 or self.state[1] <= -5:
            terminated = bool(1.0)
        if self.state[0] < 0.0:
            terminated = bool(1.0)
        if self.is_render:
            self.renderer.render_step()
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}


    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, export_video = False, export_video_filename=None):
        if export_video:
            if self.detach_video_export_process:
                self.video_export_process = Process(
                    target=self.export_video,
                    args=(export_video_filename,)
                )
                self.video_export_process.start()
            else:
                self.export_video(export_video_filename)

        # State initialization
        if self.late_start:
            # x = [X, Y, psi, vx, vy, omega, delta, D]
            self.state = np.array([70.0, 0.0, 0.0, 13.89, 0.0, 0.0, 0.0, 0.0])  # TODO: Initialization method
        else:
            self.state = np.array([0.0, 0.0, 0.0, 13.89, 0.0, 0.0, 0.0, 0.0])  # TODO: Initialization method
        self.collided_cone = None
        self.before_X = None

        # Reset environments
        super().reset(seed=seed)
        if self.is_render:
            self.renderer.reset()
            self.renderer.render_step()

        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
        
    
    def check_collision(self):
        cone_position, cone_index = self.track.get_nearlest_cone(self.state[0], return_cone_index=True)
        cone_position_rel = self.model.R(self.state[2]).T @ (cone_position - self.state[:2])

        if abs(cone_position_rel[0]) < 0.5 * self.model.L + self.cone_radius and abs(cone_position_rel[1]) < 0.5 * self.model.W + self.cone_radius:
            self.collided_cone = (cone_position, cone_index)
            print("Collision occurred")
            return True
        return False

    
    def export_video(self, filename=None):
        if filename is None:
            filename = self.video_export_location+asctime().replace(" ", "_").replace(":", "-") + ".mp4"
        if self.render_mode in {"rgb_array", "raw_data"}:
            from matplotlib.pyplot import figure, imshow
            if self.render_mode=="rgb_array":
                from matplotlib.animation import ArtistAnimation
                ArtistAnimation(
                    figure(),
                    [[imshow(arr, animated=True)] for arr in self.renderer.render_list],
                    interval=int(1000*self.dt), blit=True
                ).save(filename, writer="ffmpeg")
            if self.render_mode=="raw_data":
                from matplotlib.pyplot import gca
                from matplotlib.animation import FuncAnimation
                fig = figure(figsize=self.screen.figsize)
                ax = gca()
                track_artists = self.screen.get_track_artists(ax)
                vehicle_artists = self.screen.get_vehicle_artists(ax)
                FuncAnimation(
                    fig,
                    lambda idx: self.screen.update_vehicle_artists(
                        vehicle_artists, *self.renderer.render_list[idx]
                    ),
                    frames=len(self.renderer.render_list),
                    init_func=lambda: self.screen.reset_vehicle_artists(vehicle_artists),
                    interval=int(1000*self.dt), blit=True
                ).save(filename, writer="ffmpeg")
        else:
            raise NotImplementedError(
                "export_video() only available when render_mode is 'rgb_array' or 'raw_data'."
            )


    def render(self, mode="human"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)


    def _render(self, mode="human"):
        assert mode in self.metadata["render_modes"]

        if self.screen is None:
            self.screen = LivePlot(self.track, self.model, mode=mode)
            if mode == "human":
                self.screen.show()
        if self.clock is None:
            self.clock = Clock()

        self.screen.update(self.state, trajectories=self.trajectory, collided_cone=self.collided_cone)

        if mode == "human":
            if self.metadata["render_fps"] > 0:
                self.clock.tick(self.metadata["render_fps"])
            if self.is_X11_forwarding:
                self.screen.start_event_loop()

        elif mode in {"rgb_array", "single_rgb_array", "raw_data"}:
            return self.screen.get_data()


        return None
