import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import casadi as ca
import matplotlib.pyplot as plt


def gen_points():

    k = kernels.ConstantKernel() * kernels.RBF(length_scale=5.0, length_scale_bounds="fixed")

    Xl = np.reshape([130.0 + 60.0 * k for k in range(5)], (5, 1))
    Xr = np.reshape([100.0 + 60.0 * k for k in range(5)], (5, 1))
    Yl = 5.015 * np.ones((5, 1))
    Yr = 5.015 * np.ones((5, 1))

    # Xr = np.append(Xr, np.array([[385,]]), axis=0)
    # Yr = np.append(Yr, np.array([[4.2,]]), axis=0)
    
    gpl = GaussianProcessRegressor(kernel=k)
    gpl.fit(Xl, Yl)

    gpr = GaussianProcessRegressor(kernel=k)
    gpr.fit(Xr, Yr)

    Xstar = np.linspace(0.0, 400.0, 81)

    Ylstar = 5.0 - gpl.predict(Xstar[:, np.newaxis]).flatten()
    Yrstar = -5.0 + gpr.predict(Xstar[:, np.newaxis]).flatten()

    return Xstar, Ylstar, Yrstar


def save_file(X, Yl, Yr):

    np.savez("trackdata.npz",
        X=X, border_left=Yl, border_right=Yr,
        cone_position=np.array([100.0 + k * 30.0 for k in range(10)])
    )

    return None


def plot_interpolant(X, Yl, Yr):

    # Sl = ca.interpolant("ls", "bspline", [X], Yl)
    # Sr = ca.interpolant("rs", "bspline", [X], Yr)

    Sl = ca.interpolant("ls", "bspline", [X.tolist()], np.squeeze(Yl).tolist())
    Sr = ca.interpolant("ls", "bspline", [X.tolist()], np.squeeze(Yr).tolist())

    Xtest = np.linspace(0.0, 400.0, 1601)

    plt.plot(Xtest, Sl(Xtest).toarray())
    plt.plot(Xtest, Sr(Xtest).toarray())
    #plt.axis("equal")
    plt.show()

    return None



if __name__=="__main__":

    X, Yl, Yr = gen_points()
    plot_interpolant(X, Yl, Yr)
    save_file(X, Yl, Yr)
