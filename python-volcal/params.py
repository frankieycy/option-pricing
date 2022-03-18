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
# Ref: CGMY, Stochastic Volatility for Levy Processes
paramsCGMY = {"C": 0.002, "G": 10.781, "M": 9.230, "Y": 1.729} # converged
paramsCGMYkey = list(paramsCGMY.keys())
paramsCGMYval = list(paramsCGMY.values())
paramsCGMYbnd = ((0.001,10), (0.01,20), (0.01,20), (0.01,1.99))

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
paramsSVJJ = {'meanRevRate': 0.9069615739548296, 'correlation': -0.619659546836543, 'volOfVol': 0.3869597069187233, 'meanVar': 0.03097710828706043, 'currentVar': 0.005842956849646788, 'varJump': 0.08874498414314333, 'jumpInt': 0.2042720261322096, 'jumpMean': -0.07646950587413229, 'jumpSd': 0.11969310623322259}
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

#### rHeston: Pade Approx ######################################################
