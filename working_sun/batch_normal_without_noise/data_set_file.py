import numpy as np


def get_batch_normal_dataset():
    x_bias = -15
    x = np.linspace(-100, 100, 1000).reshape((-1, 1)) + x_bias
    y = 100 * x ** 2 + 3000 * x

    y += np.max(np.abs(y)) / 2
    # y += np.random.random(1000).reshape(x.shape) * (np.max(y) - np.min(y)) * 0.05
    return x, y


'''
[[1466250.        ], [1462250.00400801], [1458258.02404006], [1454274.06009613], [1450298.11217624], [1446330.18028038], [1442370.26440855], [1438418.36456076], [1434474.48073699], [1430538.61293726], [1426610.76116156], [1422690.92540989], [1418779.10568226], [1414875.30197866], [1410979.51429908], [1407091.74264354], [1403211.98701204], [1399340.24740456], [1395476.52382112], [1391620.81626171], [1387773.12472633], [1383933.44921498], [1380101.78972767], [1376278.14626438], [1372462.51882513], [1368654.90740991], [1364855.31201873], [1361063.73265157], [1357280.16930845], [1353504.62198936], [1349737.0906943 ], [1345977.57542327], [1342226.07617628], [1338482.59295331], [1334747.12575438], [1331019.67457948], [1327300.23942862], [1323588.82030178], [1319885.41719898], [1316190.03012021], [1312502.65906547], [1308823.30403477], [1305151.96502809], [1301488.64204545], [1297833.33508684], [1294186.04415226], [1290546.76924171], [1286915.5103552 ], [1283292.26749272], [1279677.04065427]]
'''
