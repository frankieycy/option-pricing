#### Heston ####################################################################
# Ref: Bakshi, Cao & Chen (1997)
# paramsBCC = {"meanRevRate": 2.052, "correlation": -0.711, "volOfVol": 0.587, "meanVar": 0.0138, "currentVar": 0.036}
paramsBCC = {"meanRevRate": 1.15, "correlation": -0.64, "volOfVol": 0.39, "meanVar": 0.04, "currentVar": 0.04}
paramsBCCkey = list(paramsBCC.keys())
paramsBCCval = list(paramsBCC.values())
paramsBCCbnd = ((0.01,10), (-0.99,0.99), (0.01,2), (0.01,1), (0.001,1))
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

# paramsVGL = {"C": 2, "G": 40, "M": 40}
paramsVGL = {"C": 25.131, "G": 80.086, "M": 80.006} # converged
paramsVGLkey = list(paramsVGL.keys())
paramsVGLval = list(paramsVGL.values())
paramsVGLbnd = ((0.001,50), (1.005,100), (1.005,100))

paramsMVG = {"p": 0.7, "vol1": 0.1, "drift1": -0.8, "timeChgVar1": 0.2,
             "vol2": 0.1, "drift2": 0.8, "timeChgVar2": 0.2}
paramsMVGkey = list(paramsMVG.keys())
paramsMVGval = list(paramsMVG.values())
paramsMVGbnd = ((0,1), (0.01,10), (-10,0), (0.00001,1), (0.01,10), (0,10), (0.00001,1))

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

# paramsPNCGMY = {"C": 0.002, "CRatio": 1, "G": 11.019, "M": 9.011, "Yp": 1.761, "Yn": 1.761}
paramsPNCGMY = {"C": 0.007, "CRatio": 1.311, "G": 4.436, "M": 16.234, "Yp": 1.332, "Yn": 1.526} # converged
paramsPNCGMYkey = list(paramsPNCGMY.keys())
paramsPNCGMYval = list(paramsPNCGMY.values())
paramsPNCGMYbnd = ((0.001,10), (1,5), (1.005,20), (1.005,20), (0.01,1.99), (0.01,1.99))

#### NIG #######################################################################
# Ref: CGMY, Stochastic Volatility for Levy Processes
paramsNIG = {"vol": 0.426, "drift": 0.190, "timeChgDrift": 23.761} # converged
paramsNIGkey = list(paramsNIG.keys())
paramsNIGval = list(paramsNIG.values())
paramsNIGbnd = ((0.01,1), (-1,1), (0.01,50))

#### BG ########################################################################

paramsBG = {"Ap": 10, "Am": 0.6, "Lp": 35, "Lm": 5}
paramsBGkey = list(paramsBG.keys())
paramsBGval = list(paramsBG.values())
paramsBGbnd = ((0,100), (0,100), (1,100), (1,100))

# paramsMBG = {"p": 0.6, "Ap1": 20, "Am1": 1, "Lp1": 40, "Lm1": 2,
#             "Ap2": 1, "Am2": 2, "Lp2": 3, "Lm2": 10}
paramsMBG = {"p": 0.813, "Ap1": 0.399, "Am1": 22.746, "Lp1": 15.477, "Lm1": 20.773,
             "Ap2": 75.997, "Am2": 0.714, "Lp2": 77.598, "Lm2": 1.904}
paramsMBGkey = list(paramsMBG.keys())
paramsMBGval = list(paramsMBG.values())
paramsMBGbnd = ((0,1), (0,100), (0,100), (1,2000), (0.1,2000), (0,100), (0,100), (1,2000), (0.1,2000))

#### SA ########################################################################
# Ref: CGMY, Stochastic Volatility for Levy Processes
# paramsVGSA = {"C": 22, "G": 55, "M": 80, "saMeanRevRate": 1.2, "saMean": 16, "saVol": 23}
paramsVGSA = {"C": 4.170, "G": 22.558, "M": 43.061, "saMeanRevRate": 0.046, "saMean": 17.021, "saVol": 7.348} # converged
paramsVGSAkey = list(paramsVGSA.keys())
paramsVGSAval = list(paramsVGSA.values())
paramsVGSAbnd = ((0.001,50), (1.005,100), (1.005,100), (0.01,10), (0.01,50), (0.01,50))

# paramsCGMYSA = {"C": 0.042, "CRatio": 1.187, "G": 4.397, "M": 16.402, "Yp": 1.628, "Yn": 1.782, "saMeanRevRate": 0.033, "saMean": 16.535, "saVol": 6.318}
# paramsCGMYSA = {'C': 0.2862541148423903, 'CRatio': 1.9749569272052367, 'G': 8.283259888324903, 'M': 15.764348804117436, 'Yp': 0.810318258561253, 'Yn': 0.8415294714657541, 'saMeanRevRate': 7.001221672704461, 'saMean': 0.16351428800258105, 'saVol': 1.51818474575785}
paramsCGMYSA = {"C": 0.044, "CRatio": 1.185, "G": 4.397, "M": 16.402, "Yp": 1.634, "Yn": 1.766, "saMeanRevRate": 0.028, "saMean": 16.532, "saVol": 6.318} # converged
paramsCGMYSAkey = list(paramsCGMYSA.keys())
paramsCGMYSAval = list(paramsCGMYSA.values())
paramsCGMYSAbnd = ((0.001,10), (1,5), (1.005,20), (1.005,20), (0.01,1.99), (0.01,1.99), (0.01,10), (0.01,50), (0.01,50))

paramsNIGSA = {"vol": 0.301, "drift": -5.156, "timeChgDrift": 27.149, "saMeanRevRate": 0.0006, "saMean": 16.517, "saVol": 5.871} # converged
paramsNIGSAkey = list(paramsNIGSA.keys())
paramsNIGSAval = list(paramsNIGSA.values())
paramsNIGSAbnd = ((0.01,1), (-10,10), (0.01,100), (0.00001,10), (0.01,50), (0.01,50))

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
# Ref: Gatheral, Rational Approximation of the Rough Heston Solution
paramsRHP = {"hurstExp": 0.1, "correlation": -0.6, "volOfVol": 0.3}
# paramsRHP = {"hurstExp": 0, "correlation": -0.500, "volOfVol": 0.336} # converged
paramsRHPkey = list(paramsRHP.keys())
paramsRHPval = list(paramsRHP.values())
paramsRHPbnd = ((0,0.3), (-0.99,0.99), (0.01,1))

#### Event Model ###############################################################
paramsGaussianEventJump = {"eventTime": 0, "jumpUpProb": 0.5, "jumpUpMean": 0.15, "jumpUpStd": 0.02, "jumpDnMean": -0.15, "jumpDnStd": 0.02}
paramsGaussianEventJumpkey = list(paramsGaussianEventJump.keys())
paramsGaussianEventJumpval = list(paramsGaussianEventJump.values())
paramsGaussianEventJumpbnd = ((0,1), (0.001,1), (0.001,0.2), (0.001,1), (-0.001,-0.2), (0.001,1))

paramsPointEventJump = {"eventTime": 0, "jumpProb": 0.5, "jump": 0.1}
paramsPointEventJumpkey = list(paramsPointEventJump.keys())
paramsPointEventJumpval = list(paramsPointEventJump.values())
paramsPointEventJumpbnd = ((0,1), (0.001,1), (0.001,0.2))