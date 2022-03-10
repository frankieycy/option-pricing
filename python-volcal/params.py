# Heston
paramsBCC = {"meanRevRate": 1.15, "correlation": -0.64, "volOfVol": 0.39, "meanVar": 0.04, "currentVar": 0.04}
paramsBCCkey = list(paramsBCC.keys())
paramsBCCval = list(paramsBCC.values())
paramsBCCBnd = ((0.01,10), (-0.99,0.99), (0.01,1), (0.01,1), (0.01,1))
# paramsBCCBnd = ((0,10), (-1,1), (0,10), (0.01,1), (0,1))
# Merton
paramsMER = {"vol": 0.1, "jumpInt": 0.1, "jumpMean": -0.4, "jumpSd": 0.2}
paramsMERkey = list(paramsMER.keys())
paramsMERval = list(paramsMER.values())
paramsMERBnd = ((0.01,1), (0,5), (-0.9,0.9), (0.01,2))
# paramsMERBnd = ((0,1), (0,10), (-1,1), (0,10))
# VGamma
# rHeston: Poor Man's Approx
# rHeston: Pade Approx
# SVJ
# SVJJ
