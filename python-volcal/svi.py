import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from numba import njit, float64, vectorize
from scipy.stats import norm
from scipy.special import ndtr
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from scipy.interpolate import InterpolatedUnivariateSpline, PchipInterpolator
from pricer import BlackScholesFormula, BlackScholesVega
plt.switch_backend("Agg")

#### Black-Scholes #############################################################

INVROOT2PI = 0.3989422804014327

@njit(float64(float64), fastmath=True, cache=True)
def _ndtr_jit(x):
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    g = 0.2316419

    k = 1.0 / (1.0 + g * np.abs(x))
    k2 = k * k
    k3 = k2 * k
    k4 = k3 * k
    k5 = k4 * k

    if x >= 0.0:
        c = (a1 * k + a2 * k2 + a3 * k3 + a4 * k4 + a5 * k5)
        phi = 1.0 - c * np.exp(-x*x/2.0) * INVROOT2PI
    else:
        phi = 1.0 - _ndtr_jit(-x)

    return phi

@vectorize([float64(float64)], fastmath=True, cache=True)
def ndtr_jit(x):
    return _ndtr_jit(x)

@njit
def BlackScholesFormula_jit(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType):
    # Black Scholes formula for call/put
    logMoneyness = np.log(spotPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    discountFactor = np.exp(-riskFreeRate*maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    d2 = d1-totalImpVol

    if optionType == 'call':
        return spotPrice * ndtr_jit(d1) - discountFactor * strike * ndtr_jit(d2)
    else:
        return discountFactor * strike * ndtr_jit(-d2) - spotPrice * ndtr_jit(-d1)

def BlackScholesFormula_fast(spotPrice, strike, maturity, riskFreeRate, impliedVol, optionType):
    # Black Scholes formula for call/put
    logMoneyness = np.log(spotPrice/strike)+riskFreeRate*maturity
    totalImpVol = impliedVol*np.sqrt(maturity)
    discountFactor = np.exp(-riskFreeRate*maturity)
    d1 = logMoneyness/totalImpVol+totalImpVol/2
    d2 = d1-totalImpVol

    if optionType == 'call':
        return spotPrice * ndtr(d1) - discountFactor * strike * ndtr(d2)
    else:
        return discountFactor * strike * ndtr(-d2) - spotPrice * ndtr(-d1)

#### Parametrization ###########################################################

def svi(a, b, sig, rho, m):
    # Raw-SVI parametrization
    def sviFunc(k):
        return a+b*(rho*(k-m)+np.sqrt((k-m)**2+sig**2))
    return sviFunc

@njit
def svi_jit(k, a, b, sig, rho, m):
    # Raw-SVI parametrization
    return a+b*(rho*(k-m)+np.sqrt((k-m)**2+sig**2))

def sviSkew(a, b, sig, rho, m):
    # SVI skew
    def sviSkewFunc(k):
        return b*((k-m)/sqrt((k-m)**2+sig**2)+rho)
    return sviFunc

def sviDensity(a, b, sig, rho, m):
    # SVI density
    def sviDensityFunc(k):
        D = np.sqrt((k-m)**2+sig**2)
        w0 = a+b*(rho*(k-m)+D)
        w1 = b*(rho+(k-m)/D)
        w2 = b*sig**2/D**3
        tmp1 = np.exp(-(k-w0/2)**2/(2*w0))/(2*np.sqrt(2*np.pi*w0))
        tmp2 = (1-k/(2*w0)*w1)**2-0.25*(0.25+1/w0)*w1**2+0.5*w2 # g(k)
        return 2*tmp1*tmp2
    return sviDensityFunc

def sviDensityFactor(a, b, sig, rho, m):
    # SVI density factor g(k)
    def sviDensityFactorFunc(k):
        D = np.sqrt((k-m)**2+sig**2)
        w0 = a+b*(rho*(k-m)+D)
        w1 = b*(rho+(k-m)/D)
        w2 = b*sig**2/D**3
        return (1-k/(2*w0)*w1)**2-0.25*(0.25+1/w0)*w1**2+0.5*w2
    return sviDensityFactorFunc

def sviCrossing(params1, params2):
    # Intersections & crossedness of two SVI slices
    sviLabels = ['a','b','sig','rho','m']

    # Cast to SVI dict
    if not isinstance(params1, dict):
        params1 = {sviLabels[i]: params1[i] for i in range(5)}
    if not isinstance(params2, dict):
        params2 = {sviLabels[i]: params2[i] for i in range(5)}

    a1, b1, s1, r1, m1 = params1.values() # Short-term
    a2, b2, s2, r2, m2 = params2.values() # Long-term

    # Quartic equation: q4 x^4 + q3 x^3 + q2 x^2 + q1 x + q0 = 0
    q2 = 1000000 * -2 * (-3 * b1 ** 4 * m1 ** 2 + b1 ** 2 * b2 ** 2 * m1 ** 2 + 4 * b1 ** 2 * b2 ** 2 * m1 * m2 +
              b1 ** 2 * b2 ** 2 * m2 ** 2 - 3 * b2 ** 4 * m2 ** 2 + 6 * b1 ** 4 * m1 ** 2 * r1 ** 2 +
              b1 ** 2 * b2 ** 2 * m1 ** 2 * r1 ** 2 + 4 * b1 ** 2 * b2 ** 2 * m1 * m2 * r1 ** 2 +
              b1 ** 2 * b2 ** 2 * m2 ** 2 * r1 ** 2 - 3 * b1 ** 4 * m1 ** 2 * r1 ** 4 - 6 * b1 ** 3 * b2 * m1 ** 2 * r1 * r2 -
              6 * b1 ** 3 * b2 * m1 * m2 * r1 * r2 - 6 * b1 * b2 ** 3 * m1 * m2 * r1 * r2 -
              6 * b1 * b2 ** 3 * m2 ** 2 * r1 * r2 + 6 * b1 ** 3 * b2 * m1 ** 2 * r1 ** 3 * r2 +
              6 * b1 ** 3 * b2 * m1 * m2 * r1 ** 3 * r2 + b1 ** 2 * b2 ** 2 * m1 ** 2 * r2 ** 2 +
              4 * b1 ** 2 * b2 ** 2 * m1 * m2 * r2 ** 2 + b1 ** 2 * b2 ** 2 * m2 ** 2 * r2 ** 2 + 6 * b2 ** 4 * m2 ** 2 * r2 ** 2 -
              3 * b1 ** 2 * b2 ** 2 * m1 ** 2 * r1 ** 2 * r2 ** 2 - 12 * b1 ** 2 * b2 ** 2 * m1 * m2 * r1 ** 2 * r2 ** 2 -
              3 * b1 ** 2 * b2 ** 2 * m2 ** 2 * r1 ** 2 * r2 ** 2 + 6 * b1 * b2 ** 3 * m1 * m2 * r1 * r2 ** 3 +
              6 * b1 * b2 ** 3 * m2 ** 2 * r1 * r2 ** 3 - 3 * b2 ** 4 * m2 ** 2 * r2 ** 4 -
              a1 ** 2 * (b1 ** 2 * (-1 + 3 * r1 ** 2) - 6 * b1 * b2 * r1 * r2 + b2 ** 2 * (-1 + 3 * r2 ** 2)) -
              a2 ** 2 * (b1 ** 2 * (-1 + 3 * r1 ** 2) - 6 * b1 * b2 * r1 * r2 + b2 ** 2 * (-1 + 3 * r2 ** 2)) -
              2 * a2 * (3 * b1 ** 3 * m1 * r1 * (-1 + r1 ** 2) - b1 ** 2 * b2 * (2 * m1 + m2) * (-1 +
              3 * r1 ** 2) * r2 - 3 * b2 ** 3 * m2 * r2 * (-1 + r2 ** 2) + b1 * b2 ** 2 * (m1 + 2 * m2) *
              r1 * (-1 + 3 * r2 ** 2)) + 2 * a1 * (3 * b1 ** 3 * m1 * r1 * (-1 + r1 ** 2) -
              b1 ** 2 * b2 * (2 * m1 + m2) * (-1 + 3 * r1 ** 2) * r2 - 3 * b2 ** 3 * m2 * r2 * (-1 +
              r2 ** 2) + b1 * b2 ** 2 * (m1 + 2 * m2) * r1 * (-1 + 3 * r2 ** 2) +
              a2 * (b1 ** 2 * (-1 + 3 * r1 ** 2) - 6 * b1 * b2 * r1 * r2 + b2 ** 2 * (-1 + 3 * r2 ** 2))) -
              b1 ** 4 * s1 ** 2 + b1 ** 2 * b2 ** 2 * s1 ** 2 + b1 ** 4 * r1 ** 2 * s1 ** 2 - 2 * b1 ** 3 * b2 * r1 * r2 *
              s1 ** 2 + b1 ** 2 * b2 ** 2 * r2 ** 2 * s1 ** 2 + b1 ** 2 * b2 ** 2 * s2 ** 2 - b2 ** 4 * s2 ** 2 +
              b1 ** 2 * b2 ** 2 * r1 ** 2 * s2 ** 2 - 2 * b1 * b2 ** 3 * r1 * r2 * s2 ** 2 + b2 ** 4 * r2 ** 2 * s2 ** 2)

    q4 = 1000000 * (b1 ** 4 * (-1 + r1 ** 2) ** 2 - 4 * b1 ** 3 * b2 * r1 * (-1 + r1 ** 2) * r2 -
              4 * b1 * b2 ** 3 * r1 * r2 * (-1 + r2 ** 2) + b2 ** 4 * (-1 + r2 ** 2) ** 2 +
              2 * b1 ** 2 * b2 ** 2 * (-1 - r2 ** 2 + r1 ** 2 * (-1 + 3 * r2 ** 2)))

    q0 = 1000000 * (a1 ** 4 + a2 ** 4 + b1 ** 4 * m1 ** 4 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 + b2 ** 4 * m2 ** 4 -
              2 * b1 ** 4 * m1 ** 4 * r1 ** 2 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 * r1 ** 2 + b1 ** 4 * m1 ** 4 * r1 ** 4 +
              4 * b1 ** 3 * b2 * m1 ** 3 * m2 * r1 * r2 + 4 * b1 * b2 ** 3 * m1 * m2 ** 3 * r1 * r2 -
              4 * b1 ** 3 * b2 * m1 ** 3 * m2 * r1 ** 3 * r2 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 * r2 ** 2 -
              2 * b2 ** 4 * m2 ** 4 * r2 ** 2 + 6 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 * r1 ** 2 * r2 ** 2 -
              4 * b1 * b2 ** 3 * m1 * m2 ** 3 * r1 * r2 ** 3 + b2 ** 4 * m2 ** 4 * r2 ** 4 +
              4 * a2 ** 3 * (b1 * m1 * r1 - b2 * m2 * r2) - 4 * a1 ** 3 * (a2 + b1 * m1 * r1 -
              b2 * m2 * r2) + 2 * b1 ** 4 * m1 ** 2 * s1 ** 2 - 2 * b1 ** 2 * b2 ** 2 * m2 ** 2 * s1 ** 2 -
              2 * b1 ** 4 * m1 ** 2 * r1 ** 2 * s1 ** 2 + 4 * b1 ** 3 * b2 * m1 * m2 * r1 * r2 * s1 ** 2 -
              2 * b1 ** 2 * b2 ** 2 * m2 ** 2 * r2 ** 2 * s1 ** 2 + b1 ** 4 * s1 ** 4 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * s2 ** 2 +
              2 * b2 ** 4 * m2 ** 2 * s2 ** 2 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * r1 ** 2 * s2 ** 2 +
              4 * b1 * b2 ** 3 * m1 * m2 * r1 * r2 * s2 ** 2 - 2 * b2 ** 4 * m2 ** 2 * r2 ** 2 * s2 ** 2 -
              2 * b1 ** 2 * b2 ** 2 * s1 ** 2 * s2 ** 2 + b2 ** 4 * s2 ** 4 + 4 * a2 * (b1 * m1 * r1 - b2 * m2 * r2) *
              (-2 * b1 * b2 * m1 * m2 * r1 * r2 + b1 ** 2 * (m1 ** 2 * (-1 + r1 ** 2) - s1 ** 2) +
              b2 ** 2 * (m2 ** 2 * (-1 + r2 ** 2) - s2 ** 2)) - 4 * a1 * (a2 + b1 * m1 * r1 -
              b2 * m2 * r2) * (a2 ** 2 - 2 * b1 * b2 * m1 * m2 * r1 * r2 + 2 * a2 * (b1 * m1 * r1 -
              b2 * m2 * r2) + b1 ** 2 * (m1 ** 2 * (-1 + r1 ** 2) - s1 ** 2) +
              b2 ** 2 * (m2 ** 2 * (-1 + r2 ** 2) - s2 ** 2)) + 2 * a2 ** 2 *
              (-6 * b1 * b2 * m1 * m2 * r1 * r2 + b1 ** 2 * (m1 ** 2 * (-1 + 3 * r1 ** 2) - s1 ** 2) +
              b2 ** 2 * (m2 ** 2 * (-1 + 3 * r2 ** 2) - s2 ** 2)) +
              2 * a1 ** 2 * (3 * a2 ** 2 - 6 * b1 * b2 * m1 * m2 * r1 * r2 + 6 * a2 * (b1 * m1 * r1 -
              b2 * m2 * r2) + b1 ** 2 * (m1 ** 2 * (-1 + 3 * r1 ** 2) - s1 ** 2) +
              b2 ** 2 * (m2 ** 2 * (-1 + 3 * r2 ** 2) - s2 ** 2)))

    q3 = 1000000 * -4 * (b1 ** 4 * m1 * (-1 + r1 ** 2) ** 2 - b1 ** 3 * r1 * (-1 + r1 ** 2) *
              (a1 - a2 + b2 * (3 * m1 + m2) * r2) + b2 ** 3 * (-1 + r2 ** 2) *
              ((a1 - a2) * r2 + b2 * m2 * (-1 + r2 ** 2)) + b1 * b2 ** 2 * r1 *
              (a1 - 3 * a1 * r2 ** 2 - b2 * (m1 + 3 * m2) * r2 * (-1 + r2 ** 2) +
              a2 * (-1 + 3 * r2 ** 2)) + b1 ** 2 * b2 * ((a1 - a2) * (-1 + 3 * r1 ** 2) * r2 +
              b2 * (m1 + m2) * (-1 - r2 ** 2 + r1 ** 2 * (-1 + 3 * r2 ** 2))))

    q1 = 1000000 * 4 * (-(b1 ** 4 * m1 ** 3) + b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 + b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 -
              b2 ** 4 * m2 ** 3 + 2 * b1 ** 4 * m1 ** 3 * r1 ** 2 + b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 * r1 ** 2 +
              b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 * r1 ** 2 - b1 ** 4 * m1 ** 3 * r1 ** 4 - b1 ** 3 * b2 * m1 ** 3 * r1 * r2 -
              3 * b1 ** 3 * b2 * m1 ** 2 * m2 * r1 * r2 - 3 * b1 * b2 ** 3 * m1 * m2 ** 2 * r1 * r2 - b1 * b2 ** 3 * m2 ** 3 * r1 * r2 + b1 ** 3 * b2 * m1 ** 3 * r1 ** 3 * r2 + 3 * b1 ** 3 * b2 * m1 ** 2 * m2 *
              r1 ** 3 * r2 + b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 * r2 ** 2 + b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 * r2 ** 2 +
              2 * b2 ** 4 * m2 ** 3 * r2 ** 2 - 3 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 * r1 ** 2 * r2 ** 2 -
              3 * b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 * r1 ** 2 * r2 ** 2 + 3 * b1 * b2 ** 3 * m1 * m2 ** 2 * r1 * r2 ** 3 +
              b1 * b2 ** 3 * m2 ** 3 * r1 * r2 ** 3 - b2 ** 4 * m2 ** 3 * r2 ** 4 + a1 ** 3 * (b1 * r1 - b2 * r2) +
              a2 ** 3 * (-(b1 * r1) + b2 * r2) + a2 ** 2 * (b1 ** 2 * (m1 - 3 * m1 * r1 ** 2) + 3 * b1 * b2 * (m1 + m2) * r1 * r2 + b2 ** 2 * m2 * (1 - 3 * r2 ** 2)) +
              a1 ** 2 * (b1 ** 2 * (m1 - 3 * m1 * r1 ** 2) + 3 * b1 * r1 * (-a2 + b2 * (m1 + m2) * r2) +
              b2 * (3 * a2 * r2 + b2 * (m2 - 3 * m2 * r2 ** 2))) - b1 ** 4 * m1 * s1 ** 2 +
              b1 ** 2 * b2 ** 2 * m2 * s1 ** 2 + b1 ** 4 * m1 * r1 ** 2 * s1 ** 2 - b1 ** 3 * b2 * m1 * r1 * r2 * s1 ** 2 -
              b1 ** 3 * b2 * m2 * r1 * r2 * s1 ** 2 + b1 ** 2 * b2 ** 2 * m2 * r2 ** 2 * s1 ** 2 +
              b1 ** 2 * b2 ** 2 * m1 * s2 ** 2 - b2 ** 4 * m2 * s2 ** 2 + b1 ** 2 * b2 ** 2 * m1 * r1 ** 2 * s2 ** 2 -
              b1 * b2 ** 3 * m1 * r1 * r2 * s2 ** 2 - b1 * b2 ** 3 * m2 * r1 * r2 * s2 ** 2 +
              b2 ** 4 * m2 * r2 ** 2 * s2 ** 2 + a2 * (b1 ** 2 * b2 * r2 * (m1 ** 2 * (-1 + 3 * r1 ** 2) +
              2 * m1 * m2 * (-1 + 3 * r1 ** 2) - s1 ** 2) + b1 ** 3 * r1 * (-3 * m1 ** 2 *
              (-1 + r1 ** 2) + s1 ** 2) + b2 ** 3 * r2 * (3 * m2 ** 2 * (-1 + r2 ** 2) - s2 ** 2) +
              b1 * b2 ** 2 * r1 * (m1 * m2 * (2 - 6 * r2 ** 2) + m2 ** 2 * (1 - 3 * r2 ** 2) + s2 ** 2)) +
              a1 * (3 * a2 ** 2 * (b1 * r1 - b2 * r2) + a2 * (2 * b1 ** 2 * m1 * (-1 + 3 * r1 ** 2) -
              6 * b1 * b2 * (m1 + m2) * r1 * r2 + 2 * b2 ** 2 * m2 * (-1 + 3 * r2 ** 2)) +
              b1 ** 3 * r1 * (3 * m1 ** 2 * (-1 + r1 ** 2) - s1 ** 2) + b1 ** 2 * b2 * r2 * (
              m1 * m2 * (2 - 6 * r1 ** 2) + m1 ** 2 * (1 - 3 * r1 ** 2) + s1 ** 2) +
              b1 * b2 ** 2 * r1 * (2 * m1 * m2 * (-1 + 3 * r2 ** 2) + m2 ** 2 * (-1 + 3 * r2 ** 2) -
              s2 ** 2) + b2 ** 3 * r2 * (-3 * m2 ** 2 * (-1 + r2 ** 2) + s2 ** 2)))

    # Quartic roots
    with np.errstate(divide='ignore', invalid='ignore'):

        term16 = (2 * q2 ** 3 + 27 * q3 ** 2 * q0 - 72 * q4 * q2 * q0 - 9 * q3 * q2 * q1 + 27 * q4 * q1 ** 2)
        term21 = (q2 ** 2 / 4 + 3 * q4 * q0 - 3 * q3 * q1 / 4)
        term1sq = -256 * term21 ** 3 + term16 ** 2
        term1 = np.sqrt(term1sq + 0*1j)
        term23 = (term16 + term1) ** (1/3)
        term22 = 3 * q4 * term23

        temp1 = (4 * 2 ** (1 / 3) * term21)
        temp2 = (3 * 2 ** (1 / 3) * q4)
        temp3 = q3 ** 2 / (4 * q4 ** 2) - (2 * q2) / (3 * q4)
        temp4 = temp1 / term22 + term23 / temp2

        rr = np.sqrt(temp3 + temp4)

        temp5 = q3 ** 2 / (2 * q4 ** 2) - (4 * q2) / (3 * q4)
        temp6 = (-q3 ** 3 / 4 + q4 * q3 * q2 - 2 * q4 ** 2 * q1) / (q4 ** 3)

        ee = q3 ** 2 / (2 * q4 ** 2) - (4 * q2) / (3 * q4) - (4 * 2 ** (1 / 3) * term21) / term22 - term23 / (3 * 2 ** (1 / 3) * q4) - \
            (-q3 ** 3 / 4 + q4 * q3 * q2 - 2 * q4 ** 2 * q1) / (q4 ** 3 * rr)
        dd = q3 ** 2 / (2 * q4 ** 2) - (4 * q2) / (3 * q4) - (4 * 2 ** (1 / 3) * term21) / term22 - term23 / (3 * 2 ** (1 / 3) * q4) + \
            (-q3 ** 3 / 4 + q4 * q3 * q2 - 2 * q4 ** 2 * q1) / (q4 ** 3 * rr)

        temp7 = -q3 / (4 * q4)

        # Candidate roots
        roots = np.array([
            -q3 / (4 * q4) + rr / 2 + np.sqrt(dd) / 2,
            -q3 / (4 * q4) + rr / 2 - np.sqrt(dd) / 2,
            -q3 / (4 * q4) - rr / 2 + np.sqrt(ee) / 2,
            -q3 / (4 * q4) - rr / 2 - np.sqrt(ee) / 2
        ])

    # Real roots
    kr = roots * (np.abs(np.imag(roots)) < 1e-10)
    test = lambda k: (a1 + b1 * (r1 * (k - m1) + np.sqrt((k - m1) ** 2 + s1 ** 2))) - (a2 + b2 * (r2 * (k - m2) + np.sqrt((k - m2) ** 2 + s2 ** 2)))

    idx = (np.abs(test(kr)) < 1e-10)

    roots = np.sort(np.real(kr[idx]))
    nroots = len(roots)

    # Crossedness
    cross = 0

    if nroots > 1:
        midpt = (roots[:-1]+roots[1:])/2
    else:
        midpt = []

    if nroots > 0:
        samplept = np.concatenate([[roots[0]-1], midpt, [roots[-1]+1]])
        svi1 = svi(**params1)(samplept)
        svi2 = svi(**params2)(samplept)
        cross = max(max(svi1-svi2), 0)

    return {
        'roots': roots,
        'cross': cross,
    }

@njit
def sviCrossing_jit(a1, b1, s1, r1, m1, a2, b2, s2, r2, m2):
    # Intersections & crossedness of two SVI slices

    # Quartic equation: q4 x^4 + q3 x^3 + q2 x^2 + q1 x + q0 = 0
    q2 = 1000000 * -2 * (-3 * b1 ** 4 * m1 ** 2 + b1 ** 2 * b2 ** 2 * m1 ** 2 + 4 * b1 ** 2 * b2 ** 2 * m1 * m2 +
              b1 ** 2 * b2 ** 2 * m2 ** 2 - 3 * b2 ** 4 * m2 ** 2 + 6 * b1 ** 4 * m1 ** 2 * r1 ** 2 +
              b1 ** 2 * b2 ** 2 * m1 ** 2 * r1 ** 2 + 4 * b1 ** 2 * b2 ** 2 * m1 * m2 * r1 ** 2 +
              b1 ** 2 * b2 ** 2 * m2 ** 2 * r1 ** 2 - 3 * b1 ** 4 * m1 ** 2 * r1 ** 4 - 6 * b1 ** 3 * b2 * m1 ** 2 * r1 * r2 -
              6 * b1 ** 3 * b2 * m1 * m2 * r1 * r2 - 6 * b1 * b2 ** 3 * m1 * m2 * r1 * r2 -
              6 * b1 * b2 ** 3 * m2 ** 2 * r1 * r2 + 6 * b1 ** 3 * b2 * m1 ** 2 * r1 ** 3 * r2 +
              6 * b1 ** 3 * b2 * m1 * m2 * r1 ** 3 * r2 + b1 ** 2 * b2 ** 2 * m1 ** 2 * r2 ** 2 +
              4 * b1 ** 2 * b2 ** 2 * m1 * m2 * r2 ** 2 + b1 ** 2 * b2 ** 2 * m2 ** 2 * r2 ** 2 + 6 * b2 ** 4 * m2 ** 2 * r2 ** 2 -
              3 * b1 ** 2 * b2 ** 2 * m1 ** 2 * r1 ** 2 * r2 ** 2 - 12 * b1 ** 2 * b2 ** 2 * m1 * m2 * r1 ** 2 * r2 ** 2 -
              3 * b1 ** 2 * b2 ** 2 * m2 ** 2 * r1 ** 2 * r2 ** 2 + 6 * b1 * b2 ** 3 * m1 * m2 * r1 * r2 ** 3 +
              6 * b1 * b2 ** 3 * m2 ** 2 * r1 * r2 ** 3 - 3 * b2 ** 4 * m2 ** 2 * r2 ** 4 -
              a1 ** 2 * (b1 ** 2 * (-1 + 3 * r1 ** 2) - 6 * b1 * b2 * r1 * r2 + b2 ** 2 * (-1 + 3 * r2 ** 2)) -
              a2 ** 2 * (b1 ** 2 * (-1 + 3 * r1 ** 2) - 6 * b1 * b2 * r1 * r2 + b2 ** 2 * (-1 + 3 * r2 ** 2)) -
              2 * a2 * (3 * b1 ** 3 * m1 * r1 * (-1 + r1 ** 2) - b1 ** 2 * b2 * (2 * m1 + m2) * (-1 +
              3 * r1 ** 2) * r2 - 3 * b2 ** 3 * m2 * r2 * (-1 + r2 ** 2) + b1 * b2 ** 2 * (m1 + 2 * m2) *
              r1 * (-1 + 3 * r2 ** 2)) + 2 * a1 * (3 * b1 ** 3 * m1 * r1 * (-1 + r1 ** 2) -
              b1 ** 2 * b2 * (2 * m1 + m2) * (-1 + 3 * r1 ** 2) * r2 - 3 * b2 ** 3 * m2 * r2 * (-1 +
              r2 ** 2) + b1 * b2 ** 2 * (m1 + 2 * m2) * r1 * (-1 + 3 * r2 ** 2) +
              a2 * (b1 ** 2 * (-1 + 3 * r1 ** 2) - 6 * b1 * b2 * r1 * r2 + b2 ** 2 * (-1 + 3 * r2 ** 2))) -
              b1 ** 4 * s1 ** 2 + b1 ** 2 * b2 ** 2 * s1 ** 2 + b1 ** 4 * r1 ** 2 * s1 ** 2 - 2 * b1 ** 3 * b2 * r1 * r2 *
              s1 ** 2 + b1 ** 2 * b2 ** 2 * r2 ** 2 * s1 ** 2 + b1 ** 2 * b2 ** 2 * s2 ** 2 - b2 ** 4 * s2 ** 2 +
              b1 ** 2 * b2 ** 2 * r1 ** 2 * s2 ** 2 - 2 * b1 * b2 ** 3 * r1 * r2 * s2 ** 2 + b2 ** 4 * r2 ** 2 * s2 ** 2)

    q4 = 1000000 * (b1 ** 4 * (-1 + r1 ** 2) ** 2 - 4 * b1 ** 3 * b2 * r1 * (-1 + r1 ** 2) * r2 -
              4 * b1 * b2 ** 3 * r1 * r2 * (-1 + r2 ** 2) + b2 ** 4 * (-1 + r2 ** 2) ** 2 +
              2 * b1 ** 2 * b2 ** 2 * (-1 - r2 ** 2 + r1 ** 2 * (-1 + 3 * r2 ** 2)))

    q0 = 1000000 * (a1 ** 4 + a2 ** 4 + b1 ** 4 * m1 ** 4 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 + b2 ** 4 * m2 ** 4 -
              2 * b1 ** 4 * m1 ** 4 * r1 ** 2 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 * r1 ** 2 + b1 ** 4 * m1 ** 4 * r1 ** 4 +
              4 * b1 ** 3 * b2 * m1 ** 3 * m2 * r1 * r2 + 4 * b1 * b2 ** 3 * m1 * m2 ** 3 * r1 * r2 -
              4 * b1 ** 3 * b2 * m1 ** 3 * m2 * r1 ** 3 * r2 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 * r2 ** 2 -
              2 * b2 ** 4 * m2 ** 4 * r2 ** 2 + 6 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 ** 2 * r1 ** 2 * r2 ** 2 -
              4 * b1 * b2 ** 3 * m1 * m2 ** 3 * r1 * r2 ** 3 + b2 ** 4 * m2 ** 4 * r2 ** 4 +
              4 * a2 ** 3 * (b1 * m1 * r1 - b2 * m2 * r2) - 4 * a1 ** 3 * (a2 + b1 * m1 * r1 -
              b2 * m2 * r2) + 2 * b1 ** 4 * m1 ** 2 * s1 ** 2 - 2 * b1 ** 2 * b2 ** 2 * m2 ** 2 * s1 ** 2 -
              2 * b1 ** 4 * m1 ** 2 * r1 ** 2 * s1 ** 2 + 4 * b1 ** 3 * b2 * m1 * m2 * r1 * r2 * s1 ** 2 -
              2 * b1 ** 2 * b2 ** 2 * m2 ** 2 * r2 ** 2 * s1 ** 2 + b1 ** 4 * s1 ** 4 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * s2 ** 2 +
              2 * b2 ** 4 * m2 ** 2 * s2 ** 2 - 2 * b1 ** 2 * b2 ** 2 * m1 ** 2 * r1 ** 2 * s2 ** 2 +
              4 * b1 * b2 ** 3 * m1 * m2 * r1 * r2 * s2 ** 2 - 2 * b2 ** 4 * m2 ** 2 * r2 ** 2 * s2 ** 2 -
              2 * b1 ** 2 * b2 ** 2 * s1 ** 2 * s2 ** 2 + b2 ** 4 * s2 ** 4 + 4 * a2 * (b1 * m1 * r1 - b2 * m2 * r2) *
              (-2 * b1 * b2 * m1 * m2 * r1 * r2 + b1 ** 2 * (m1 ** 2 * (-1 + r1 ** 2) - s1 ** 2) +
              b2 ** 2 * (m2 ** 2 * (-1 + r2 ** 2) - s2 ** 2)) - 4 * a1 * (a2 + b1 * m1 * r1 -
              b2 * m2 * r2) * (a2 ** 2 - 2 * b1 * b2 * m1 * m2 * r1 * r2 + 2 * a2 * (b1 * m1 * r1 -
              b2 * m2 * r2) + b1 ** 2 * (m1 ** 2 * (-1 + r1 ** 2) - s1 ** 2) +
              b2 ** 2 * (m2 ** 2 * (-1 + r2 ** 2) - s2 ** 2)) + 2 * a2 ** 2 *
              (-6 * b1 * b2 * m1 * m2 * r1 * r2 + b1 ** 2 * (m1 ** 2 * (-1 + 3 * r1 ** 2) - s1 ** 2) +
              b2 ** 2 * (m2 ** 2 * (-1 + 3 * r2 ** 2) - s2 ** 2)) +
              2 * a1 ** 2 * (3 * a2 ** 2 - 6 * b1 * b2 * m1 * m2 * r1 * r2 + 6 * a2 * (b1 * m1 * r1 -
              b2 * m2 * r2) + b1 ** 2 * (m1 ** 2 * (-1 + 3 * r1 ** 2) - s1 ** 2) +
              b2 ** 2 * (m2 ** 2 * (-1 + 3 * r2 ** 2) - s2 ** 2)))

    q3 = 1000000 * -4 * (b1 ** 4 * m1 * (-1 + r1 ** 2) ** 2 - b1 ** 3 * r1 * (-1 + r1 ** 2) *
              (a1 - a2 + b2 * (3 * m1 + m2) * r2) + b2 ** 3 * (-1 + r2 ** 2) *
              ((a1 - a2) * r2 + b2 * m2 * (-1 + r2 ** 2)) + b1 * b2 ** 2 * r1 *
              (a1 - 3 * a1 * r2 ** 2 - b2 * (m1 + 3 * m2) * r2 * (-1 + r2 ** 2) +
              a2 * (-1 + 3 * r2 ** 2)) + b1 ** 2 * b2 * ((a1 - a2) * (-1 + 3 * r1 ** 2) * r2 +
              b2 * (m1 + m2) * (-1 - r2 ** 2 + r1 ** 2 * (-1 + 3 * r2 ** 2))))

    q1 = 1000000 * 4 * (-(b1 ** 4 * m1 ** 3) + b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 + b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 -
              b2 ** 4 * m2 ** 3 + 2 * b1 ** 4 * m1 ** 3 * r1 ** 2 + b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 * r1 ** 2 +
              b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 * r1 ** 2 - b1 ** 4 * m1 ** 3 * r1 ** 4 - b1 ** 3 * b2 * m1 ** 3 * r1 * r2 -
              3 * b1 ** 3 * b2 * m1 ** 2 * m2 * r1 * r2 - 3 * b1 * b2 ** 3 * m1 * m2 ** 2 * r1 * r2 - b1 * b2 ** 3 * m2 ** 3 * r1 * r2 + b1 ** 3 * b2 * m1 ** 3 * r1 ** 3 * r2 + 3 * b1 ** 3 * b2 * m1 ** 2 * m2 *
              r1 ** 3 * r2 + b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 * r2 ** 2 + b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 * r2 ** 2 +
              2 * b2 ** 4 * m2 ** 3 * r2 ** 2 - 3 * b1 ** 2 * b2 ** 2 * m1 ** 2 * m2 * r1 ** 2 * r2 ** 2 -
              3 * b1 ** 2 * b2 ** 2 * m1 * m2 ** 2 * r1 ** 2 * r2 ** 2 + 3 * b1 * b2 ** 3 * m1 * m2 ** 2 * r1 * r2 ** 3 +
              b1 * b2 ** 3 * m2 ** 3 * r1 * r2 ** 3 - b2 ** 4 * m2 ** 3 * r2 ** 4 + a1 ** 3 * (b1 * r1 - b2 * r2) +
              a2 ** 3 * (-(b1 * r1) + b2 * r2) + a2 ** 2 * (b1 ** 2 * (m1 - 3 * m1 * r1 ** 2) + 3 * b1 * b2 * (m1 + m2) * r1 * r2 + b2 ** 2 * m2 * (1 - 3 * r2 ** 2)) +
              a1 ** 2 * (b1 ** 2 * (m1 - 3 * m1 * r1 ** 2) + 3 * b1 * r1 * (-a2 + b2 * (m1 + m2) * r2) +
              b2 * (3 * a2 * r2 + b2 * (m2 - 3 * m2 * r2 ** 2))) - b1 ** 4 * m1 * s1 ** 2 +
              b1 ** 2 * b2 ** 2 * m2 * s1 ** 2 + b1 ** 4 * m1 * r1 ** 2 * s1 ** 2 - b1 ** 3 * b2 * m1 * r1 * r2 * s1 ** 2 -
              b1 ** 3 * b2 * m2 * r1 * r2 * s1 ** 2 + b1 ** 2 * b2 ** 2 * m2 * r2 ** 2 * s1 ** 2 +
              b1 ** 2 * b2 ** 2 * m1 * s2 ** 2 - b2 ** 4 * m2 * s2 ** 2 + b1 ** 2 * b2 ** 2 * m1 * r1 ** 2 * s2 ** 2 -
              b1 * b2 ** 3 * m1 * r1 * r2 * s2 ** 2 - b1 * b2 ** 3 * m2 * r1 * r2 * s2 ** 2 +
              b2 ** 4 * m2 * r2 ** 2 * s2 ** 2 + a2 * (b1 ** 2 * b2 * r2 * (m1 ** 2 * (-1 + 3 * r1 ** 2) +
              2 * m1 * m2 * (-1 + 3 * r1 ** 2) - s1 ** 2) + b1 ** 3 * r1 * (-3 * m1 ** 2 *
              (-1 + r1 ** 2) + s1 ** 2) + b2 ** 3 * r2 * (3 * m2 ** 2 * (-1 + r2 ** 2) - s2 ** 2) +
              b1 * b2 ** 2 * r1 * (m1 * m2 * (2 - 6 * r2 ** 2) + m2 ** 2 * (1 - 3 * r2 ** 2) + s2 ** 2)) +
              a1 * (3 * a2 ** 2 * (b1 * r1 - b2 * r2) + a2 * (2 * b1 ** 2 * m1 * (-1 + 3 * r1 ** 2) -
              6 * b1 * b2 * (m1 + m2) * r1 * r2 + 2 * b2 ** 2 * m2 * (-1 + 3 * r2 ** 2)) +
              b1 ** 3 * r1 * (3 * m1 ** 2 * (-1 + r1 ** 2) - s1 ** 2) + b1 ** 2 * b2 * r2 * (
              m1 * m2 * (2 - 6 * r1 ** 2) + m1 ** 2 * (1 - 3 * r1 ** 2) + s1 ** 2) +
              b1 * b2 ** 2 * r1 * (2 * m1 * m2 * (-1 + 3 * r2 ** 2) + m2 ** 2 * (-1 + 3 * r2 ** 2) -
              s2 ** 2) + b2 ** 3 * r2 * (-3 * m2 ** 2 * (-1 + r2 ** 2) + s2 ** 2)))

    # Quartic roots
    # with np.errstate(divide='ignore', invalid='ignore'):

    if np.abs(q4) < 1e-15:
        q4 = 1e-15

    term16 = (2 * q2 ** 3 + 27 * q3 ** 2 * q0 - 72 * q4 * q2 * q0 - 9 * q3 * q2 * q1 + 27 * q4 * q1 ** 2)
    term21 = (q2 ** 2 / 4 + 3 * q4 * q0 - 3 * q3 * q1 / 4)
    term1sq = -256 * term21 ** 3 + term16 ** 2
    term1 = np.sqrt(term1sq + 0*1j)
    term23 = (term16 + term1) ** (1/3)
    term22 = 3 * q4 * term23

    temp1 = (4 * 2 ** (1 / 3) * term21)
    temp2 = (3 * 2 ** (1 / 3) * q4)
    temp3 = q3 ** 2 / (4 * q4 ** 2) - (2 * q2) / (3 * q4)
    temp4 = temp1 / term22 + term23 / temp2

    rr = np.sqrt(temp3 + temp4)

    temp5 = q3 ** 2 / (2 * q4 ** 2) - (4 * q2) / (3 * q4)
    temp6 = (-q3 ** 3 / 4 + q4 * q3 * q2 - 2 * q4 ** 2 * q1) / (q4 ** 3)

    ee = q3 ** 2 / (2 * q4 ** 2) - (4 * q2) / (3 * q4) - (4 * 2 ** (1 / 3) * term21) / term22 - term23 / (3 * 2 ** (1 / 3) * q4) - \
        (-q3 ** 3 / 4 + q4 * q3 * q2 - 2 * q4 ** 2 * q1) / (q4 ** 3 * rr)
    dd = q3 ** 2 / (2 * q4 ** 2) - (4 * q2) / (3 * q4) - (4 * 2 ** (1 / 3) * term21) / term22 - term23 / (3 * 2 ** (1 / 3) * q4) + \
        (-q3 ** 3 / 4 + q4 * q3 * q2 - 2 * q4 ** 2 * q1) / (q4 ** 3 * rr)

    temp7 = -q3 / (4 * q4)

    # Candidate roots
    roots = np.array([
        -q3 / (4 * q4) + rr / 2 + np.sqrt(dd) / 2,
        -q3 / (4 * q4) + rr / 2 - np.sqrt(dd) / 2,
        -q3 / (4 * q4) - rr / 2 + np.sqrt(ee) / 2,
        -q3 / (4 * q4) - rr / 2 - np.sqrt(ee) / 2
    ])

    # Real roots
    kr = roots * (np.abs(np.imag(roots)) < 1e-10)
    test = lambda k: (a1 + b1 * (r1 * (k - m1) + np.sqrt((k - m1) ** 2 + s1 ** 2))) - (a2 + b2 * (r2 * (k - m2) + np.sqrt((k - m2) ** 2 + s2 ** 2)))

    idx = (np.abs(test(kr)) < 1e-10)

    roots = np.sort(np.real(kr[idx]))
    nroots = len(roots)

    # Crossedness
    cross = 0

    if nroots > 1:
        midpt = list((roots[:-1]+roots[1:])/2)
    else:
        midpt = [np.float64(x) for x in range(0)]

    if nroots > 0:
        samplept = np.array([roots[0]-1] + midpt + [roots[-1]+1])
        svi1 = svi_jit(samplept,a1,b1,s1,r1,m1)
        svi2 = svi_jit(samplept,a2,b2,s2,r2,m2)
        cross = max(max(svi1-svi2), 0)

    return cross

def sviSqrt(w0, rho, eta):
    # Sqrt-SVI parametrization
    def sviFunc(k):
        sk = eta/np.sqrt(w0) # Sqrt skew decay
        return w0/2*(1+rho*sk*k+np.sqrt((sk*k+rho)**2+1-rho**2))
    return sviFunc

@njit
def sviSqrt_jit(k, w0, rho, eta):
    # Sqrt-SVI parametrization
    sk = eta/np.sqrt(w0) # Sqrt skew decay
    return w0/2*(1+rho*sk*k+np.sqrt((sk*k+rho)**2+1-rho**2))

def sviPowerLaw(w0, rho, eta, gam):
    # PowerLaw-SVI parametrization
    def sviFunc(k):
        sk = eta/(w0**gam*(1+w0)**(1-gam)) # PowerLaw skew decay
        return w0/2*(1+rho*sk*k+np.sqrt((sk*k+rho)**2+1-rho**2))
    return sviFunc

@njit
def sviPowerLaw_jit(k, w0, rho, eta, gam):
    sk = eta/(w0**gam*(1+w0)**(1-gam)) # PowerLaw skew decay
    return w0/2*(1+rho*sk*k+np.sqrt((sk*k+rho)**2+1-rho**2))

def sviHeston(w0, rho, eta, lda):
    # Heston-SVI parametrization
    def sviFunc(k):
        sk = eta*(1-(1-np.exp(-lda*w0))/(lda*w0))/(lda*w0) # Heston skew decay
        return w0/2*(1+rho*sk*k+np.sqrt((sk*k+rho)**2+1-rho**2))
    return sviFunc

@njit
def sviHeston_jit(k, w0, rho, eta, lda):
    sk = eta*(1-(1-np.exp(-lda*w0))/(lda*w0))/(lda*w0) # Heston skew decay
    return w0/2*(1+rho*sk*k+np.sqrt((sk*k+rho)**2+1-rho**2))

def esviPowerLaw(w0, eta, gam, rho0, rho1, wmax, a):
    # PowerLaw-eSVI parametrization
    def sviFunc(k):
        sk = eta/(w0**gam*(1+w0)**(1-gam)) # PowerLaw skew decay
        rho = rho0-(rho0-rho1)*(w0/wmax)**a # PowerLaw corr decay
        return w0/2*(1+rho*sk*k+np.sqrt((sk*k+rho)**2+1-rho**2))
    return sviFunc

@njit
def esviPowerLaw_jit(k, w0, eta, gam, rho0, rho1, wmax, a):
    sk = eta/(w0**gam*(1+w0)**(1-gam)) # PowerLaw skew decay
    rho = rho0-(rho0-rho1)*(w0/wmax)**a # PowerLaw corr decay
    return w0/2*(1+rho*sk*k+np.sqrt((sk*k+rho)**2+1-rho**2))

#### Arbitrage Check ###########################################################

def CalendarArbLoss(params1, params2):
    # Penalty for calendar spread arbitrage (across two slices)
    # sviCrx = sviCrossing(params1, params2)
    # loss = sviCrx['cross']
    # return loss
    return sviCrossing_jit(*params1, *params2)

def ButterflyArbLoss(params):
    # Penalty for butterfly spread arbitrage (single slice)
    g = sviDensityFactor(**params)
    opt = minimize_scalar(g, bounds=(-2,2), method="Bounded")
    loss = -min(opt.fun, 0)
    return loss

def GenVogtButterflyArbitrage(params0=(0,0.1,0.4,0.3,0.3), penalty=(8,1,1)):
    # Generate SVI params that gives butterfly arbitrage
    # Vogt smile: (-0.0410,0.1331,0.4153,0.3060,0.3586)
    k = np.arange(-1.5,1.5,1e-4)
    def objective(params):
        gk = sviDensityFactor(*params)(k)
        loss1 = penalty[0]*min(gk) # negative density
        loss2 = penalty[1]*sum(abs(gk[1:]-gk[:-1])) # smoothness
        loss3 = penalty[2]*sum((params-params0)**2) # deviation from params0
        print(f"params: {params} loss1: {loss1} loss2: {loss2} loss3: {loss3}")
        return loss1+loss2+loss3
    bounds = ((-10,10),(0,10),(0,10),(-0.99,0.99),(-10,10))
    opt = minimize(objective, x0=params0, bounds=bounds)
    return opt.x

#### Surface Fitting ###########################################################
# Return fit with raw-SVI parametrization, in total implied variance w=sig^2*T

def FitSimpleSVI(df, sviGuess=None, initParamsMode=0):
    # Fit Simple SVI to each slice independently
    # Columns: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    df = df.dropna()
    df = df[(df['Bid']>0)&(df['Ask']>0)]
    Texp = df["Texp"].unique()

    fit = dict()

    for i,T in enumerate(Texp):
        dfT = df[df["Texp"]==T]
        k = np.log(dfT["Strike"]/dfT["Fwd"])
        bid = dfT["Bid"]
        ask = dfT["Ask"]
        midVar = (bid**2+ask**2)/2
        sprdVar = (ask**2-bid**2)/2

        k = k.to_numpy()
        midVar = midVar.to_numpy()
        sprdVar = sprdVar.to_numpy()

        @njit
        def varToL2Loss(sviVar): # Fast loss computation!
            return np.sum(((sviVar-midVar)/sprdVar)**2)

        def loss(params): # L2 loss
            # sviVar = svi(*params)(k)
            sviVar = svi_jit(k,*params)
            # return sum((sviVar-midVar)**2)
            # return sum(((sviVar-midVar)/sprdVar)**2)
            return varToL2Loss(sviVar)

        # Initial params
        if sviGuess is None:
            if initParamsMode == 0: # Fixed a
                params0 = (0, 0.1, 0.1, -0.7, 0)
            elif initParamsMode == 1: # Dynamic a
                params0 = (np.mean(midVar), 0.1, 0.1, -0.7, 0)
            elif initParamsMode == 2: # Based on prev slice
                if i == 0: params0 = (np.mean(midVar), 0.1, 0.1, -0.7, 0)
                else: params0 = fit[Texp[i-1]]
        else:
            params0 = sviGuess.iloc[i]

        opt = minimize(loss, x0=params0, bounds=((-10,10),(0,10),(0,10),(-0.99,0.99),(-10,10)))
        fit[T] = opt.x * np.array([T,T,1,1,1])

        err = np.sqrt(np.sqrt(opt.fun/len(dfT))*np.mean(sprdVar))*100 # Error in vol points

        print(f'i={i} T={np.round(T,4)} err={np.round(err,4)}% fit={opt.x}')

    fit = pd.DataFrame(fit).T
    fit.columns = ['a','b','sig','rho','m']

    return fit

def FitArbFreeSimpleSVI(df, sviGuess=None, initParamsMode=0, cArbPenalty=10000, l2Weight=None):
    # Fit Simple SVI to each slice guaranteeing no static arbitrage
    # Optimization w/o sviGuess is unstable under L2 loss in price!
    # df Columns: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    # sviGuess Columns: "a","b","sig","rho","m"
    df = df.dropna()
    df = df[(df['Bid']>0)&(df['Ask']>0)]
    Texp = df["Texp"].unique()
    Nexp = len(Texp)

    if sviGuess is None:
        fit = pd.DataFrame(index=Texp, columns=['a','b','sig','rho','m'])
    else:
        fit = sviGuess.copy() # Parametrization for w=sig^2*T

    for i,T in enumerate(Texp):
        dfT = df[df["Texp"]==T]
        F = dfT["Fwd"]
        K = dfT["Strike"]
        k = np.log(dfT["Strike"]/dfT["Fwd"])
        bid = dfT["Bid"]
        ask = dfT["Ask"]
        mid = (bid+ask)/2
        callMid = dfT["CallMid"]

        F = F.to_numpy()
        K = K.to_numpy()
        k = k.to_numpy()
        callMid = callMid.to_numpy()

        if l2Weight == "Vega":
            w = BlackScholesVega(F,K,T,0,mid,'call')
        elif l2Weight == "Gaussian":
            w = norm.pdf(k/0.5)
        elif l2Weight == "BidAsk":
            callBid = BlackScholesFormula(F,K,T,0,bid,'call')
            callAsk = BlackScholesFormula(F,K,T,0,ask,'call')
            w = 1/(callAsk-callBid)
        else:
            w = 1

        @njit
        def prxToL2Loss(callSvi): # Fast loss computation!
            return np.sum(w*(callSvi-callMid)**2)

        def l2Loss(params): # L2 loss
            # sviVar = svi(*params)(k)/T
            sviVar = svi_jit(k,*params)/T
            callSvi = BlackScholesFormula_jit(F,K,T,0,np.sqrt(np.abs(sviVar)),'call')
            # callSvi = BlackScholesFormula_fast(F,K,T,0,np.sqrt(np.abs(sviVar)),'call')
            # loss = sum(w*(callSvi-callMid)**2)
            loss = prxToL2Loss(callSvi)
            return loss

        def caLoss(params): # Calendar arb loss
            loss = 0
            if i == 0:
                a,b,s,r,m = params
                minVar = a+b*s*np.sqrt(1-r**2)
                loss += min(100, np.exp(-1/minVar))
            if i > 0:
                loss += CalendarArbLoss(fit.iloc[i-1], params)
            # if i < Nexp-1:
            #     loss += CalendarArbLoss(params, fit.iloc[i+1])
            return loss

        # Normalizer
        if sviGuess is None: # Constant normalizer is unstable!
            l2loss0 = l2Loss((0, 0.1, 0.1, -0.7, 0))
        else:
            l2loss0 = l2Loss(sviGuess.iloc[i])

        def loss(params):
            loss1 = l2Loss(params)/l2loss0
            loss2 = caLoss(params)*cArbPenalty
            return loss1 + loss2

        # Initial params
        if sviGuess is None:
            if i == 0: params0 = (0, 0.1, 0.1, -0.7, 0)
            else: params0 = fit.iloc[i-1]
        else:
            if initParamsMode == 0: # Use guess for small T
                if i == 0 or T < 0.05: params0 = sviGuess.iloc[i]
                else: params0 = fit.iloc[i-1]
            elif initParamsMode == 1: # Use guess for all T
                params0 = sviGuess.iloc[i]

        opt = minimize(loss, x0=params0, bounds=((-10,10),(0,10),(0,10),(-0.99,0.99),(-10,10)))
        fit.iloc[i] = opt.x

        print(f'i={i} T={np.round(T,4)} loss={np.round(opt.fun,4)} fit={opt.x}')

    return fit

def FitSqrtSVI(df, sviGuess=None, Tcut=0.2):
    # Fit Sqrt-Surface SVI to all slices
    # Columns: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    # Ref: Gatheral/Jacquier, Arbitrage-free SVI Volatility Surfaces
    df = df.dropna()
    df = df[(df['Bid']>0)&(df['Ask']>0)]
    Texp = df["Texp"].unique()

    fit = dict()

    k = np.log(df["Strike"]/df["Fwd"])
    k = k.to_numpy()
    bid = df["Bid"].to_numpy()
    ask = df["Ask"].to_numpy()
    midVar = (bid**2+ask**2)/2
    sprdVar = (ask**2-bid**2)/2

    w0 = np.zeros(len(df))
    T0 = df["Texp"].to_numpy()

    for T in Texp:
        i = (T0==T)
        kT = k[i]
        vT = midVar[i]
        ntm = (kT>-0.05)&(kT<0.05)
        spline = InterpolatedUnivariateSpline(kT[ntm], vT[ntm])
        w0[i] = spline(0).item()*T # ATM total variance

    @njit
    def varToL2Loss(sviVar): # Fast loss computation!
        if Tcut: # Fit to longer-term slices
            return np.sum((((sviVar-midVar)/sprdVar)**2)[T0>=Tcut])
        else: # Fit to all slices
            return np.sum(((sviVar-midVar)/sprdVar)**2)

    def loss(params): # L2 loss
        sviVar = sviSqrt(w0,*params)(k)/T0
        # sviVar = sviSqrt_jit(k,w0,*params)/T0
        # return sum((sviVar-midVar)**2)
        # return sum(((sviVar-midVar)/sprdVar)**2)
        if Tcut: # Fit to longer-term slices
            return sum((((sviVar-midVar)/sprdVar)**2)[T0>=Tcut])
        else: # Fit to all slices
            return sum(((sviVar-midVar)/sprdVar)**2)
        # return varToL2Loss(sviVar)

    # Initial params
    if sviGuess is None:
        params0 = (-0.7, 1)
    else:
        params0 = sviGuess

    opt = minimize(loss, x0=params0, bounds=((-0.99,0.99),(-10,10)))
    # opt = differential_evolution(loss, bounds=((-0.99,0.99),(-10,10)))

    print(f'loss={np.round(opt.fun,4)} fit={opt.x}')

    # Cast to raw-SVI
    w0 = np.unique(w0)
    rho, eta = opt.x
    sk = eta/np.sqrt(w0)
    a = w0/2*(1-rho**2)
    b = w0/2*sk
    s = np.sqrt(1-rho**2)/sk
    r = rho
    m = -rho/sk

    fit = {'a':a, 'b':b, 'sig':s, 'rho':r, 'm':m}
    fit = pd.DataFrame(fit)
    fit.index = Texp

    return fit

def FitSurfaceSVI(df, sviGuess=None, skewKernel='PowerLaw', Tcut=0.2):
    # Fit Surface SVI to all slices
    # Columns: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    # Ref: Gatheral/Jacquier, Arbitrage-free SVI Volatility Surfaces
    df = df.dropna()
    df = df[(df['Bid']>0)&(df['Ask']>0)]
    Texp = df["Texp"].unique()

    fit = dict()

    k = np.log(df["Strike"]/df["Fwd"])
    k = k.to_numpy()
    bid = df["Bid"].to_numpy()
    ask = df["Ask"].to_numpy()
    midVar = (bid**2+ask**2)/2
    sprdVar = (ask**2-bid**2)/2

    w0 = np.zeros(len(df))
    T0 = df["Texp"].to_numpy()

    for T in Texp:
        i = (T0==T)
        kT = k[i]
        vT = midVar[i]
        ntm = (kT>-0.05)&(kT<0.05)
        spline = InterpolatedUnivariateSpline(kT[ntm], vT[ntm])
        w0[i] = spline(0).item()*T # ATM total variance

    # SVI function & initial params
    if skewKernel == 'Sqrt':
        sviFunc = sviSqrt
        # sviFunc = sviSqrt_jit
        skFunc = lambda w0,eta: eta/np.sqrt(w0)
        params0 = (-0.7, 1) if sviGuess is None else sviGuess
        bounds0 = ((-0.99, 0.99), (-10, 10))
    elif skewKernel == 'PowerLaw':
        sviFunc = sviPowerLaw
        # sviFunc = sviPowerLaw_jit
        skFunc = lambda w0,eta,gam: eta/(w0**gam*(1+w0)**(1-gam))
        params0 = (-0.7, 1, 0.3) if sviGuess is None else sviGuess
        # bounds0 = ((-0.99, 0.99), (-10, 10), (0.01, 0.5))
        bounds0 = ((-0.99, 0.99), (-10, 10), (0.01, 1)) # modified: gam
    elif skewKernel == 'Heston':
        sviFunc = sviHeston
        # sviFunc = sviHeston_jit
        skFunc = lambda w0,eta,lda: eta*(1-(1-np.exp(-lda*w0))/(lda*w0))/(lda*w0)
        params0 = (-0.7, 50, 100) if sviGuess is None else sviGuess
        bounds0 = ((-0.99, 0.99), (-1000, 1000), (0, 1000))

    @njit
    def varToL2Loss(sviVar): # Fast loss computation!
        if Tcut: # Fit to longer-term slices
            return np.sum((((sviVar-midVar)/sprdVar)**2)[T0>=Tcut])
        else: # Fit to all slices
            return np.sum(((sviVar-midVar)/sprdVar)**2)

    def loss(params): # L2 loss
        sviVar = sviFunc(w0,*params)(k)/T0
        # sviVar = sviFunc(k,w0,*params)/T0
        # return sum((sviVar-midVar)**2)
        # return sum(((sviVar-midVar)/sprdVar)**2)
        if Tcut: # Fit to longer-term slices
            return sum((((sviVar-midVar)/sprdVar)**2)[T0>=Tcut])
        else: # Fit to all slices
            return sum(((sviVar-midVar)/sprdVar)**2)
        # return varToL2Loss(sviVar)

    opt = minimize(loss, x0=params0, bounds=bounds0)
    # opt = differential_evolution(loss, bounds=bounds0)

    print(f'loss={np.round(opt.fun,4)} fit={opt.x}')

    # Cast to raw-SVI
    w0 = np.unique(w0)
    rho, *par = opt.x
    sk = skFunc(w0,*par)
    a = w0/2*(1-rho**2)
    b = w0/2*sk
    s = np.sqrt(1-rho**2)/sk
    r = rho
    m = -rho/sk

    fit = {'a':a, 'b':b, 'sig':s, 'rho':r, 'm':m}
    fit = pd.DataFrame(fit)
    fit.index = Texp

    return fit

def FitExtendedSurfaceSVI(df, sviGuess=None, Tcut=0.2):
    # Fit Extended Surface SVI to all slices, with PowerLaw skew & corr decay
    # Columns: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    # Ref: Hendriks/Martini, The Extended SSVI Volatility Surface
    df = df.dropna()
    df = df[(df['Bid']>0)&(df['Ask']>0)]
    Texp = df["Texp"].unique()

    fit = dict()

    k = np.log(df["Strike"]/df["Fwd"])
    k = k.to_numpy()
    bid = df["Bid"].to_numpy()
    ask = df["Ask"].to_numpy()
    midVar = (bid**2+ask**2)/2
    sprdVar = (ask**2-bid**2)/2

    w0 = np.zeros(len(df))
    T0 = df["Texp"].to_numpy()

    for T in Texp:
        i = (T0==T)
        kT = k[i]
        vT = midVar[i]
        ntm = (kT>-0.05)&(kT<0.05)
        spline = InterpolatedUnivariateSpline(kT[ntm], vT[ntm])
        w0[i] = spline(0).item()*T # ATM total variance

    # SVI function & initial params
    sviFunc = esviPowerLaw
    # sviFunc = esviPowerLaw_jit
    skFunc = lambda w0,eta,gam: eta/(w0**gam*(1+w0)**(1-gam))
    rhoFunc = lambda w0,rho0,rho1,wmax,a: rho0-(rho0-rho1)*(w0/wmax)**a
    params0 = (1, 0.3, -0.7, -0.8, 2, 0.5) if sviGuess is None else sviGuess
    # bounds0 = ((-10, 10), (0.01, 0.5), (-0.99, 0.99), (-0.99, 0.99), (0.01, 10), (0, 10))
    bounds0 = ((-10, 10), (0.01, 1), (-0.99, 0.99), (-0.99, 0.99), (0.01, 10), (0, 1)) # modified: gam,a

    # @njit
    # def varToL2Loss(sviVar): # Fast loss computation!
    #     if Tcut: # Fit to longer-term slices
    #         return np.sum((((sviVar-midVar)/sprdVar)**2)[T0>=Tcut])
    #     else: # Fit to all slices
    #         return np.sum(((sviVar-midVar)/sprdVar)**2)

    def loss(params): # L2 loss
        sviVar = sviFunc(w0,*params)(k)/T0
        # sviVar = sviFunc(k,w0,*params)/T0
        # return sum((sviVar-midVar)**2)
        # return sum(((sviVar-midVar)/sprdVar)**2)
        if Tcut: # Fit to longer-term slices
            return sum((((sviVar-midVar)/sprdVar)**2)[T0>=Tcut])
        else: # Fit to all slices
            return sum(((sviVar-midVar)/sprdVar)**2)
        # return varToL2Loss(sviVar)

    opt = minimize(loss, x0=params0, bounds=bounds0)
    # opt = differential_evolution(loss, bounds=bounds0)

    print(f'loss={np.round(opt.fun,4)} fit={opt.x}')

    # Cast to raw-SVI
    w0 = np.unique(w0)
    eta, gam, *par = opt.x
    sk = skFunc(w0,eta,gam)
    rho = rhoFunc(w0,*par)
    a = w0/2*(1-rho**2)
    b = w0/2*sk
    s = np.sqrt(1-rho**2)/sk
    r = rho
    m = -rho/sk

    fit = {'a':a, 'b':b, 'sig':s, 'rho':r, 'm':m}
    fit = pd.DataFrame(fit)
    fit.index = Texp

    return fit

def FitArbFreeSimpleSVIWithSimSeed(df, initParamsMode=0, cArbPenalty=10000):
    # Fit Simple SVI to each slice guaranteeing no static arbitrage with Simple-SVI seed
    # Columns: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    guess = FitSimpleSVI(df)
    fit = FitArbFreeSimpleSVI(df, guess, initParamsMode, cArbPenalty)
    return fit

def FitArbFreeSimpleSVIWithSqrtSeed(df, initParamsMode=0, cArbPenalty=10000, Tcut=0.2):
    # Fit Simple SVI to each slice guaranteeing no static arbitrage with Sqrt-SVI seed
    # Columns: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    guess = FitSqrtSVI(df, Tcut=Tcut)
    fit = FitArbFreeSimpleSVI(df, guess, initParamsMode, cArbPenalty)
    return fit

#### Other Parametrizations ####################################################

def sviParamsToJW(df):
    # Cast SVI params to jump-wing representation
    # Columns: "a","b","sig","rho","m"; Index: maturity
    a = df['a']
    b = df['b']
    sig = df['sig']
    rho = df['rho']
    m = df['m']
    T = df.index

    w0 = a+b*(np.sqrt(m**2+sig**2)-rho*m)
    bb = b/np.sqrt(w0)

    v = w0/T
    vmin = (a+b*sig*np.sqrt(1-rho**2))/T
    psi = bb*(rho-m/np.sqrt(m**2+sig**2))/2
    p = bb*(1-rho)
    c = bb*(1+rho)

    fit = {'v':v, 'vmin':vmin, 'psi':psi, 'p':p, 'c':c}
    fit = pd.DataFrame(fit)
    fit.index = T

    return fit

def jwParamsToSVI(df):
    # Cast jump-wing params to SVI representation
    # Columns: "v","vmin","psi","p","c"; Index: maturity
    v = df['v']
    vmin = df['vmin']
    psi = df['psi']
    p = df['p']
    c = df['c']
    T = df.index

    ww = np.sqrt(v*T)
    bb = (p+c)/2

    b = bb*ww
    rho = 1-p/bb

    beta = rho-2*psi/bb
    alpha = np.sign(beta)*np.sqrt(1/beta**2-1)

    m = (v-vmin)*T/(b*(-rho+np.sign(alpha)*np.sqrt(1+alpha**2)-alpha*np.sqrt(1-rho**2)))
    sig = alpha*m
    a = vmin*T-b*sig*np.sqrt(1-rho**2)

    fit = {'a':a, 'b':b, 'sig':sig, 'rho':rho, 'm':m}
    fit = pd.DataFrame(fit)
    fit.index = T

    return fit

#### Surface Construction ######################################################

def SVIVolSurface(fit):
    # SVI vol surface inter/extrapolation
    # fit: term-structure of raw-SVI parametrization
    # ivSurface applies to vector k & T
    Texp = fit.index.to_numpy()
    def ivSurface(k,T):
        grid = np.array([svi(*fit.loc[t])(k) for t in Texp])
        surf = np.array([PchipInterpolator(Texp,grid[:,i])(T) for i in range(len(k))])
        surf = np.sqrt(surf/T).T
        return surf
    return ivSurface

#### ATM Term-Structure ########################################################

def SVIAtmTermStructure(df, type='vol'):
    # SVI ATM term-structure, including vol/skew/curv
    # fit: term-structure of raw-SVI parametrization
    a = df['a']
    b = df['b']
    sig = df['sig']
    rho = df['rho']
    m = df['m']
    T = df.index

    D = np.sqrt(m**2+sig**2)

    # total var w=sig^2*T
    atm = a+b*(-rho*m+D)
    skew = b*(rho-m/D)
    curv = b*sig**2/D**3
    p = b*(1-rho)
    c = b*(1+rho)

    if type == 'var':
        atm /= T
        skew /= T
        curv /= T
        p /= T
        c /= T
    elif type == 'vol':
        A = 2*np.sqrt(atm*T)
        curv = (curv-skew**2/(2*atm))/A
        skew = skew/A
        atm = np.sqrt(atm/T)
        p = 0 # scale as 1/sqrt(k) tending to 0
        c = 0

    ts = {'atm':atm, 'skew':skew, 'curv':curv, 'p':p, 'c':c}
    ts = pd.DataFrame(ts)
    ts.index = T

    return ts
