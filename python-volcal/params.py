#### Heston ####################################################################
paramsBCC = {"meanRevRate": 1.15, "correlation": -0.64, "volOfVol": 0.39, "meanVar": 0.04, "currentVar": 0.04}
paramsBCCkey = list(paramsBCC.keys())
paramsBCCval = list(paramsBCC.values())
paramsBCCbnd = ((0.01,10), (-0.99,0.99), (0.01,1), (0.01,1), (0.01,1))
# paramsBCCbnd = ((0,10), (-1,1), (0,10), (0.01,1), (0,1))
#### Merton ####################################################################
paramsMER = {"vol": 0.1, "jumpInt": 0.1, "jumpMean": -0.4, "jumpSd": 0.2}
paramsMERkey = list(paramsMER.keys())
paramsMERval = list(paramsMER.values())
paramsMERbnd = ((0.01,1), (0,5), (-0.9,0.9), (0.01,2))
# paramsMERbnd = ((0,1), (0,10), (-1,1), (0,10))
#### VGamma ####################################################################
paramsVG = {"vol": 0.095, "drift": -0.096, "timeChgVar": 0.288}
paramsVGkey = list(paramsVG.keys())
paramsVGval = list(paramsVG.values())
paramsVGbnd = ((0.01,1), (-1,1), (0.01,1))
#### SVJ #######################################################################
paramsSVJ = {"meanRevRate": 0.704, "correlation": -0.530, "volOfVol": 0.872, "meanVar": 0.061, "currentVar": 0.01, "jumpInt": 0.006, "jumpMean": -0.900, "jumpSd": 0.821}
paramsSVJkey = list(paramsSVJ.keys())
paramsSVJval = list(paramsSVJ.values())
paramsSVJbnd = ((0.01,10), (-0.99,0), (0.01,1), (0.01,1), (0.01,1), (0,2), (-0.9,0.5), (0.01,2))
#### rHeston: Poor Man's Approx ################################################
#### rHeston: Pade Approx ######################################################
