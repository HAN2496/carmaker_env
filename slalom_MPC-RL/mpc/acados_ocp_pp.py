import casadi as ca
from numpy import array, zeros, ones, hstack, diag, eye, sqrt
from scipy.linalg import block_diag
from acados_template import AcadosOcp
from environment.utils import RK4



def acados_ocp_pp(vehicle_model, track, config):

    ocp = AcadosOcp()

    ## CasADi Model

    # set up state x
    theta = ca.MX.sym("theta")
    ec = ca.MX.sym("ec")
    epsi = ca.MX.sym("epsi")
    vx = ca.MX.sym("vx")
    vy = ca.MX.sym("vy")
    omega = ca.MX.sym("omega")
    delta = ca.MX.sym("delta")
    D = ca.MX.sym("D")
    x = ca.vertcat(theta, ec, epsi, vx, vy, omega, delta, D)
    ocp.model.x = x
    nx = ocp.model.x.size1()

    # controls
    ddelta = ca.MX.sym("ddelta")
    u = ca.vertcat(ddelta)
    ocp.model.u = u
    nu = ocp.model.u.size1()

    ny = nx + nu
    ny_e = nx

    # xdot
    thetadot = ca.MX.sym("thetadot")
    ecdot = ca.MX.sym("ecdot")
    epsidot = ca.MX.sym("epsidot")
    vxdot = ca.MX.sym("vxdot")
    vydot = ca.MX.sym("vydot")
    omegadot = ca.MX.sym("omegadot")
    deltadot = ca.MX.sym("deltadot")
    Ddot = ca.MX.sym("Ddot")
    xdot = ca.vertcat(thetadot, ecdot, epsidot, vxdot, vydot, omegadot, deltadot, Ddot)
    ocp.model.xdot = xdot
 

    # algebraic variables
    z = ca.vertcat([])
    ocp.model.z = z
    nz = ocp.model.z.size1()


    # model parameters
    thetaref = ca.MX.sym("thetaref")
    qtheta = ca.MX.sym("qtheta")
    qc = ca.MX.sym("qc")
    ecref = ca.MX.sym("ecref")
    ddelta_before = ca.MX.sym("ddelta_before")
    dt = ca.MX.sym("dt")
    
    p = ca.vertcat(ddelta_before, qtheta, thetaref, qc, ecref, dt)
    ocp.model.p = p
    np = ocp.model.p.size1()
    ocp.parameter_values = zeros(np)


    # dynamics
    f_expl = dt * ca.vertcat(
        *vehicle_model.f_pp(
            theta, ec, epsi, vx, vy, omega,
            delta, D, ddelta, 0.0
        )
    ) # ddelta is the only control var.

    xnew = [x] # [theta, ec, epsi, vx, vy, omega, delta, D]
    for _ in range(config.mpc.sim_method_num_steps):
        xnew.append(
            RK4(
                lambda xk: dt * ca.vertcat(
                    *vehicle_model.f_pp(
                        xk[0], xk[1], xk[2], xk[3], xk[4], xk[5],
                        delta, D, ddelta, 0.0,
                    )
                ),
                config.mpc.dt/config.mpc.sim_method_num_steps, xnew[-1]
            )
        )
    disc_dyn = xnew[-1]

    ocp.model.f_impl_expr = xdot - f_expl
    ocp.model.f_expl_expr = f_expl
    ocp.model.disc_dyn_expr = disc_dyn
    

    # Set path constraint h
    safe_distance = config.vehicle.W*3/5 + config.env.cone_radius # might be wrong

    border_left = ca.interpolant("bound_l_s", "bspline", [track.X.tolist()], track.border_left.tolist())
    border_right = ca.interpolant("bound_l_s", "bspline", [track.X.tolist()], track.border_right.tolist())
    
    Y = ec
    ubY = border_left(theta) - safe_distance
    lbY = border_right(theta) + safe_distance
    normalized_Y = (Y - (ubY + lbY)/2) / ((ubY - lbY)/2) # |Y-중앙선| < 폭/2
    ocp.model.con_h_expr = ca.vertcat(normalized_Y) 

    nh = ocp.model.con_h_expr.size1()
    nsh = nh


    # Set cost
    ocp.model.cost_expr_ext_cost = qc*(ec - ecref)**2 + 1e-3*ddelta*ddelta + 1e-0*(ddelta - ddelta_before)**2
    ocp.model.cost_expr_ext_cost_e = qc*(ec - ecref)**2 - qtheta*(theta - thetaref)

    ocp.model.name = "Path_parametric_MPC"


    # set constraints

    ocp.constraints.x0 = zeros(nx)

    ocp.constraints.lbx = array([0])
    ocp.constraints.ubx = array([0])
    ocp.constraints.idxbx = array([nx-1])
    nsbx = ocp.constraints.idxbx.shape[0]

    ocp.constraints.lbu = array([config.vehicle.ddelta_min])
    ocp.constraints.ubu = array([config.vehicle.ddelta_max])
    ocp.constraints.idxbu = array(range(nu))

    ocp.constraints.lsbx = zeros([nsbx])
    ocp.constraints.usbx = zeros([nsbx])
    ocp.constraints.idxsbx = array(range(nsbx))
    
    ocp.constraints.lh = array([-1])
    ocp.constraints.uh = array([1])
    ocp.constraints.lsh = zeros(nsh)
    ocp.constraints.ush = zeros(nsh)
    ocp.constraints.idxsh = array(range(nsh))


    # set cost
    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ns = nsh + nsbx
    ocp.cost.zl = 100 * ones((ns,))
    ocp.cost.zu = 100 * ones((ns,))
    ocp.cost.Zl = 1 * ones((ns,))
    ocp.cost.Zu = 1 * ones((ns,))


    return ocp