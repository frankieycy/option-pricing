#### Heston ####################################################################
# Ref: Bakshi, Cao & Chen (1997)
paramsBCC = {"meanRevRate": 1.15, "correlation": -0.64, "volOfVol": 0.39, "meanVar": 0.04, "currentVar": 0.04}
paramsBCCkey = list(paramsBCC.keys())
paramsBCCval = list(paramsBCC.values())
paramsBCCbnd = ((0.01,10), (-0.99,0.99), (0.01,2), (0.01,1), (0.01,1))
# paramsBCCbnd = ((0,10), (-1,1), (0,10), (0.01,1), (0,1))

#### Merton ####################################################################
paramsMER = {"vol": 0.1, "jumpInt": 0.1, "jumpMean": -0.4, "jumpSd": 0.2}
paramsMERkey = list(paramsMER.keys())
paramsMERval = list(paramsMER.values())
paramsMERbnd = ((0.01,1), (0,5), (-0.9,0.9), (0.01,2))
# paramsMERbnd = ((0,1), (0,10), (-1,1), (0,10))

#### VGamma ####################################################################
# Ref: Madan, The Variance Gamma Process and Option Pricing
# paramsVG = {"vol": 0.12, "drift": -0.14, "timeChgVar": 0.17}
paramsVG = {"vol": 0.095, "drift": -0.096, "timeChgVar": 0.288} # converged
paramsVGkey = list(paramsVG.keys())
paramsVGval = list(paramsVG.values())
paramsVGbnd = ((0.01,1), (-1,1), (0.01,1))

#### CGMY ######################################################################
# Ref: CGMY, The Fine Structure of Asset Returns: An Empirical Investigation
# paramsCGMY = {"C": 0.005, "G": 10, "M": 10, "Y": 1.7}
paramsCGMY = {"C": 0.002, "G": 11.019, "M": 9.011, "Y": 1.761} # converged
paramsCGMYkey = list(paramsCGMY.keys())
paramsCGMYval = list(paramsCGMY.values())
paramsCGMYbnd = ((0.001,10), (1.005,20), (1.005,20), (0.01,1.99))

# paramsECGMY = {"vol": 0.1, "C": 0.005, "G": 10, "M": 10, "Y": 1.7}
paramsECGMY = {"vol": 0.003, "C": 0.002, "G": 10.970, "M": 8.961, "Y": 1.762} # converged
paramsECGMYkey = list(paramsECGMY.keys())
paramsECGMYval = list(paramsECGMY.values())
paramsECGMYbnd = ((0,1), (0.001,10), (1.005,20), (1.005,20), (0.01,1.99))

#### SVJ #######################################################################
# Ref: Gatheral, Volatility Workshop VW2.pdf
# Empirical fits:
#         lda  eta  rho   vbar  ldaJ alp   del
# BCC SVJ 2.03 0.38 -0.57 0.04  0.59 -0.05 0.07
# M   SVJ 1.0  0.8  -0.7  0.04  0.5  -0.15 0
# DPS SVJ 3.99 0.27 -0.79 0.014 0.11 -0.12 0.15
# JG  SVJ 0.54 0.3  -0.7  0.044 0.13 -0.12 0.10
# paramsSVJ = {"meanRevRate": 0.704, "correlation": -0.530, "volOfVol": 0.872, "meanVar": 0.061, "currentVar": 0.01, "jumpInt": 0.006, "jumpMean": -0.900, "jumpSd": 0.821}
# paramsSVJ = {"meanRevRate": 0.54, "correlation": -0.70, "volOfVol": 0.30, "meanVar": 0.04, "currentVar": 0.04, "jumpInt": 0.10, "jumpMean": -0.10, "jumpSd": 0.10}
paramsSVJ = {'meanRevRate': 0.467, 'correlation': -0.587, 'volOfVol': 0.478, 'meanVar': 0.077, 'currentVar': 0.006, 'jumpInt': 0.137, 'jumpMean': -0.094, 'jumpSd': 0.158} # converged
paramsSVJkey = list(paramsSVJ.keys())
paramsSVJval = list(paramsSVJ.values())
# paramsSVJbnd = ((0.01,10), (-0.99,0), (0.01,1), (0.01,1), (0.01,1), (0,2), (-0.9,0.5), (0.01,2))
paramsSVJbnd = ((0.01,5), (-0.99,0), (0.1,1), (0.005,1), (0.005,1), (0,2), (-0.5,0.5), (0.01,0.5))

#### SVJJ ######################################################################
# Ref: Gatheral, Volatility Workshop VW2.pdf
# paramsSVJJ = {'meanRevRate': 0.467, 'correlation': -0.587, 'volOfVol': 0.478, 'meanVar': 0.077, 'currentVar': 0.006, 'varJump': 0.01, 'jumpInt': 0.137, 'jumpMean': -0.094, 'jumpSd': 0.158}
paramsSVJJ = {'meanRevRate': 1.3156204239409885, 'correlation': -0.5873772827168262, 'volOfVol': 0.454336340244502, 'meanVar': 0.027919165337448457, 'currentVar': 0.0058258349276720574, 'varJump': 0.0973422097494231, 'jumpInt': 0.17492864086366464, 'jumpMean': -0.0819577367385789, 'jumpSd': 0.12655156751767221}
paramsSVJJkey = list(paramsSVJJ.keys())
paramsSVJJval = list(paramsSVJJ.values())
paramsSVJJbnd = ((0.01,5), (-0.99,0), (0.1,1), (0.005,1), (0.005,1), (0.001,1), (0,2), (-0.5,0.5), (0.01,0.5))

#### rHeston: Poor Man's Approx ################################################
# Ref: Gatheral, Roughening Heston
# paramsRHPM = {"hurstExp": 0.05, "correlation": -0.67, "volOfVol": 0.41, "currentVar": 0.01}
paramsRHPM = {"hurstExp": 0.05, "correlation": -0.520, "volOfVol": 0.216, "currentVar": 0.012} # converged
paramsRHPMkey = list(paramsRHPM.keys())
paramsRHPMval = list(paramsRHPM.values())
paramsRHPMbnd = ((0,0.3), (-0.99,0.99), (0.01,1), (0.01,1))

#### rHeston: Poor Man's Approx Modified #######################################
# Ref: Gatheral, Roughening Heston
paramsRHPMM = {"hurstExp": 0.1, "meanRevRate": 1.15, "correlation": -0.64, "volOfVol": 0.39, "meanVar": 0.04, "currentVar": 0.04}
paramsRHPMMkey = list(paramsRHPMM.keys())
paramsRHPMMval = list(paramsRHPMM.values())
paramsRHPMMbnd = ((0,0.3), (0.01,10), (-0.99,0.99), (0.01,2), (0.01,1), (0.01,1))

#### rHeston: Pade Approx ######################################################
