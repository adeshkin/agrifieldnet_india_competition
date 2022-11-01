# NDVI SRI RENDVI ARI - https://journals-crea.4science.it/index.php/asr/article/view/1463
# others - https://sentinel-hub.com/develop/documentation/eo_products/Sentinel2EOproducts
def NDVI(x):
    # (NIR - RED) / (NIR + RED)
    # (B08 - B04) / (B08 + B04)
    return (x['B08'] - x['B04']) / (x['B08'] + x['B04'])


def SRI(x):
    # NIR / RED
    # B08 / B04
    return x['B08'] / x['B04']


def RENDVI(x):
    # (B06 - B05) / (B06 + B05)
    return (x['B06'] - x['B05']) / (x['B06'] + x['B05'])


def ARI(x):
    # 1.0 / B03 - 1.0 / B05
    return (1 / x['B03']) - (1 / x['B05'])


def SAVI(x):
    # (B08 - B04) / (B08 + B04 + L) * (1.0 + L)
    L = 0.428
    return (x['B08'] - x['B04']) / (x['B08'] + x['B04'] + L) * (1.0 + L)


def MSI(x):
    # B11 / B08
    return x['B11'] / x['B08']


def MCARI(x):
    # ((B05 - B04) - 0.2 * (B05 - B03)) * (B05 / B04)
    return ((x['B05'] - x['B04']) - 0.2 * (x['B05'] - x['B03'])) * (x['B05'] / x['B04'])


def MARI(x):
    # ((1.0 / B03) - (1.0 / B05)) * B07
    return ((1.0 / x['B03']) - (1.0 / x['B05'])) * x['B07']


def GNDVI(x):
    # (B08 - B03) / (B08 + B03)
    return (x['B08'] - x['B03']) / (x['B08'] + x['B03'])


def EVI(x):
    # 2.5 * (B08 - B04) / ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0)
    return 2.5 * (x['B08'] - x['B04']) / ((x['B08'] + 6.0 * x['B04'] - 7.5 * x['B02']) + 1.0)


def EVI2(x):
    # 2.4 * (B08 - B04) / (B08 + B04 + 1.0)
    return 2.4 * (x['B08'] - x['B04']) / (x['B08'] + x['B04'] + 1.0)


def NDMI(x):
    # (B08 - B11) / (B08 + B11)
    return (x['B08'] - x['B11']) / (x['B08'] + x['B11'])


def NDWI(x):
    # (B03 - B08) / (B03 + B08)
    return (x['B03'] - x['B08']) / (x['B03'] + x['B08'])