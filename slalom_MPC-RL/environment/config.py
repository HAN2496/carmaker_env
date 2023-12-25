'''
Racing environment configuration file
'''
from .utils import Parameters, dataclass
from dataclasses import field
from casadi import tanh
from numpy import array



@dataclass
class Environment_Parameters(Parameters):

    integrator_type : str = "RK45"  # ["RK45", "RK23", "DOP853", "radau", "BDF", "LSODA"]
    dt : float = 0.02
    cone_radius : float = 0.01
    sim_method_num_steps : int = 3
    is_constant_track_border : bool = True
    render_fps : int = 60  # Set 0 if fps limit to be disabled.
    is_X11_forwarding : bool = True  # Set True if the GUI is opened via X11 forwarding (e. g. remote ssh, docker)
    detach_video_export_process : bool = True
    video_export_location : str = "./video/"
    late_start : bool = False # if True, starts right before the first cone.
    Kv : float = 5.3 # P gain for longitudinal velocity feedback control.
    vref : float = 13.89 # longitudinal velocity reference (50km/h) while X \in [0m, 70m]
    

@dataclass
class Vehicle_Parameters(Parameters):

    Cm1 : float = 0.287
    Cm2 : float = 0.0545
    Cr0 : float = 0.0518
    Cr2 : float = 3.5e-4
    Cr3 : float = 5.0

    # Static params
    m   : float = 2283.36
    Iz  : float = 4650
    lf  : float = 0.985
    lr  : float = 1.970
    L   : float = 4.940
    W   : float = 1.975
    
    # Lateral dynamics
    Br  : float = 15.0693841577676
    Cr  : float = 1.44323056767426
    Dr  : float = 9326.44447146823

    CLr : float = 198686.5
    CLf : float = 166478.6046833368

    Bf  : float = 13.163204541541
    Cf  : float = 1.45027155063759
    Df  : float = 9032.91975562235

    # Longitudinal dynamics
    rw : float = 0.36875
    mu : float = 0.9
    gravity : float = 9.81

    rolling_coef_fx : float = 0.00685303788632066 
    rolling_coef_rx : float = 0.00364373791962774

    # rolling_coef_fx : float = 0.012 
    # rolling_coef_rx : float = 0.006

    # input bounds
    tau_min : float = -10000.0
    tau_max : float =  3728.0
    delta_min : float = -0.573
    delta_max : float = 0.573
    dtau_min : float = -9000
    dtau_max : float =  9000
    ddelta_min : float = -1.0
    ddelta_max : float =  1.0

    # rear steering mapping
    deltar_map_vx = array([0, 4, 8, 12, 16, 22, 28, 34, 42, 48, 58, 70, 90, 120, 160, 200]) / 3.6
    deltar_map_ratio = array([0, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.2, 0, 0.3, 0.3, 0.35, 0.4, 0.45, 0.5])


@dataclass
class MPC_Parameters(Parameters):

    dt : float = 1 #0.01
    N  : int   = 150

    Q : list = field(default_factory=lambda:
        [ 1e-1, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-3, 5e-3 ]
    )
    R : list = field(default_factory=lambda:
        [1e-3, 5e-3]
    )
    qtheta  : float = 1e-4
    qc     : float = 1e-1

    integrator_type : str = "ERK"  # ["ERK", "IRK", "DISCRETE"]
    sim_method_num_stages : int = 4
    sim_method_num_steps  : int = 3
    qp_solver : str = "PARTIAL_CONDENSING_HPIPM"
    qp_solver_warm_start : int = 1
    qp_solver_iter_max   : int = 50
    hpipm_mode : str = "BALANCE"
    hessian_approx  : str = "GAUSS_NEWTON"
    nlp_solver_type : str = "SQP_RTI"
    nlp_solver_max_iter : int = 200
    nlp_solver_step_length : float = 0.8
    globalization : str = "MERIT_BACKTRACKING"
    levenberg_marquardt: float = 0.0
    tol : float = 1e-4
    print_level : int = 0

@dataclass
class RL_Parameters(Parameters):
    h : int = 100

@dataclass
class Config(Parameters):

    env : Environment_Parameters = Environment_Parameters()
    vehicle : Vehicle_Parameters = Vehicle_Parameters()
    mpc : MPC_Parameters = MPC_Parameters()
    rl : RL_Parameters = RL_Parameters()
