from casadi import sin, cos, tanh, atan, atan2, fabs, sign, fmax, fmin, sqrt, interpolant
from numpy import array



class VehicleModel:

    def __init__(self, config):

        self.m                  = config.vehicle.m
        self.Iz                 = config.vehicle.Iz
        self.lf                 = config.vehicle.lf
        self.lr                 = config.vehicle.lr

        self.rw                 = config.vehicle.rw 
        self.mu                 = config.vehicle.mu
        # self.drag_coef_fx       = config.vehicle.drag_coef_fx
        self.rolling_coef_fx    = config.vehicle.rolling_coef_fx
        # self.drag_coef_rx       = config.vehicle.drag_coef_rx
        self.rolling_coef_rx    = config.vehicle.rolling_coef_rx
        
        self.Bf                 = config.vehicle.Bf
        self.Cf                 = config.vehicle.Cf
        self.Df                 = config.vehicle.Df

        self.Br                 = config.vehicle.Br
        self.Cr                 = config.vehicle.Cr
        self.Dr                 = config.vehicle.Dr

        self.CLr                = config.vehicle.CLr
        self.CLf                = config.vehicle.CLf
        
        self.L                  = 4.97
        self.W                  = 1.662

        self.tau_min    = config.vehicle.tau_min
        self.tau_max    = config.vehicle.tau_max
        self.delta_min  = config.vehicle.delta_min
        self.delta_max  = config.vehicle.delta_max
        self.dtau_min   = config.vehicle.dtau_min
        self.dtau_max   = config.vehicle.dtau_max
        self.ddelta_min = config.vehicle.ddelta_min
        self.ddelta_max = config.vehicle.ddelta_max

        self.deltar_map_vx = config.vehicle.deltar_map_vx
        self.deltar_map_ratio = config.vehicle.deltar_map_ratio


    def get_tire_slip_angle(self, vx, vy, omega, delta):

        vxnew = fmax(vx, 0.1)

        deltaf = delta
        deltar = self.delta_rear_mapping(vx, delta)


        alphaf = deltaf - atan2(vy + self.lf * omega, vxnew)
        alphar = deltar - atan2(vy - self.lr * omega, vxnew)

        return alphaf, alphar

    
    def get_longitudinal_tire_force(self, tau):

        F_rolling = self.mu * self.m * 9.81
        
        # Ffx = 28/40*fmin(tau, -113.4225)/self.rw  - self.rolling_coef_fx*F_rolling  
        # Frx = fmax(tau,-113.4225)/self.rw + 12/40*fmin(tau, -113.4225)/self.rw - self.rolling_coef_rx*F_rolling

        # Consider only 'Trq_Drive' and the rolling friction. Omit 'Trq_Brake'. 
        Ffx = 0 - self.rolling_coef_fx*F_rolling
        Frx = tau/self.rw - self.rolling_coef_rx*F_rolling

        return Ffx, Frx


    def get_lateral_tire_force(self, vx, vy, omega, delta):

        alphaf, alphar = self.get_tire_slip_angle(vx, vy, omega, delta)

        Ffy = self.Df * sin(self.Cf * atan(self.Bf * alphaf))
        Fry = self.Dr * sin(self.Cr * atan(self.Br * alphar))

        return Ffy, Fry

    def delta_rear_mapping(self, vx, delta):
        mapping = interpolant('mapping', 'bspline', [self.deltar_map_vx.tolist()], self.deltar_map_ratio.tolist())
        deltar = delta*mapping(vx)

        return deltar

    def f(self, X, Y, psi, vx, vy, omega, delta, tau, ddelta, dtau):
        '''
        Contunuons-time dynamics model.
        Arguments
        ---------
        X: X-axis position of the Vehicle on the global coordinates.
        Y: Y-axis position of the Vehicle on the global coordinates.
        psi: Heading angle of the Vehicle on the global coordinates.
        vx: Longitudinal velocity of the vehicle.
        vy: Lateral velocity of the vehicle.
        omega: Turning rate of the vehicle.
        delta: Steering angle.
        tau: Torque of the powertrain.
        ddelta: Steering angle change rate.
        dtau: Torque change rate. - NOT USED.
        '''

        Ffy, Fry = self.get_lateral_tire_force(vx, vy, omega, delta)
        Ffx, Frx = self.get_longitudinal_tire_force(tau)
        
        deltaf = delta
        deltar = self.delta_rear_mapping(vx, delta)
        
        dx = vx * cos(psi) - vy * sin(psi)
        dy = vx * sin(psi) + vy * cos(psi)
        dpsi = omega
        dvx = (-Ffy*sin(deltaf) + Frx*cos(deltar)- Fry*sin(deltar)) / self.m + vy * omega
        dvy = (Ffy*cos(deltaf) + Frx*sin(deltar) + Fry*cos(deltar)) / self.m - vx * omega
        domega = (Ffy*self.lf*cos(deltaf) - Frx*self.lr*sin(deltar) - Fry*self.lr*cos(deltar)) / self.Iz

        return dx, dy, dpsi, dvx, dvy, domega, ddelta, dtau


    def f_pp(self, theta, ec, epsi, vx, vy, omega, delta, tau, ddelta, dtau):
        '''
        Contunuons-time dynamics model for Path parametric MPC.
        Arguments
        ---------
        theta: Arc length of the vehicle.
        ec: Contouring error of the vehicle.
        epsi: Heading angle error of the vehicle.
        vx: Longitudinal velocity of the vehicle.
        vy: Lateral velocity of the vehicle.
        omega: Turning rate of the vehicle.
        delta: Steering angle.
        tau: Torque of the powertrain.
        ddelta: Steering angle change rate.
        dtau: Torque change rate.
        '''

        Ffy, Fry = self.get_lateral_tire_force(vx, vy, omega, delta)
        Ffx, Frx = self.get_longitudinal_tire_force(tau)
        
        deltaf = delta
        deltar = self.delta_rear_mapping(vx, delta)
        
        dtheta = vx*cos(epsi) - vy*sin(epsi)
        dec = vx*sin(epsi) + vy*cos(epsi)
        depsi = omega
        dvx = (-Ffy*sin(deltaf) + Frx*cos(deltar)- Fry*sin(deltar)) / self.m + vy * omega
        dvy = (Ffy*cos(deltaf) + Frx*sin(deltar) + Fry*cos(deltar)) / self.m - vx * omega
        domega = (Ffy*self.lf*cos(deltaf) - Frx*self.lr*sin(deltar) - Fry*self.lr*cos(deltar)) / self.Iz

        return dtheta, dec, depsi, dvx, dvy, domega, ddelta, dtau


    def footprint(self, x=0.0, y=0.0, psi=0.0):
        vertices = self.R(psi) @ array([
            [self.L, self.L, -self.L, -self.L],
            [self.W, -self.W, -self.W, self.W],
        ]) / 2 + array([[x], [y]])
        return vertices[0, :], vertices[1, :]
    

    def R(self, psi=0.0):
        cos_psi = cos(psi)
        sin_psi = sin(psi)
        return array([
            [cos_psi, -sin_psi],
            [sin_psi,  cos_psi]
        ])


    @property
    def safe_distance(self):
        #return ((self.W/2)**2 + (self.L/2)**2)**0.5
        return (self.W/2) * 1.5
