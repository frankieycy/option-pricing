import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
plt.switch_backend("Agg")

#### Parametrization ###########################################################

def svi(a, b, sig, rho, m):
    def sviFunc(k):
        return a+b*(rho*(k-m)+np.sqrt((k-m)**2+sig**2))
    return sviFunc

def sviSkew(a, b, sig, rho, m):
    def sviSkewFunc(k):
        return b*((k-m)/sqrt((k-m)**2+sig**2)+rho)
    return sviFunc

def sviDensity(a, b, sig, rho, m):
    def sviDensityFunc(k):
        D = np.sqrt((k-m)**2+sig**2)
        w0 = a+b*(rho*(k-m)+D)
        w1 = b*(rho+(k-m)/D)
        w2 = b*sig**2/D**3
        tmp1 = np.exp(-(k-w0/2)**2/(2*w0))/(2*np.sqrt(2*np.pi*w0))
        tmp2 = (1-k/(2*w0)*w1)**2-0.25*(0.25+1/w0)*w1**2+0.5*w2
        return 2*tmp1*tmp2
    return sviDensityFunc

def sviDensityFactor(a, b, sig, rho, m):
    def sviDensityFactorFunc(k):
        D = np.sqrt((k-m)**2+sig**2)
        w0 = a+b*(rho*(k-m)+D)
        w1 = b*(rho+(k-m)/D)
        w2 = b*sig**2/D**3
        return (1-k/(2*w0)*w1)**2-0.25*(0.25+1/w0)*w1**2+0.5*w2
    return sviDensityFactorFunc

def sviCrossing(params1, params2):
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

#### Arbitrage Check ###########################################################

def CalendarArbLoss(params1, params2):
    # Calendar spread arbitrage across two slices
    sviCrx = sviCrossing(params1, params2)
    loss = sviCrx['cross']
    return loss

def ButterflyArbLoss(params):
    # Butterfly spread arbitrage of a slice
    g = sviDensityFactor(**params)
    opt = minimize_scalar(g, bounds=(-2,2))
    loss = -min(opt.fun, 0)
    return loss

#### Surface Fitting ###########################################################
