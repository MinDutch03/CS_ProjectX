import opendatasets as od
import zipfile

# download data from kaggle
od.download(
    'https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification', force=True)

# unzip file
Dataset = "Adience-benchmark-gender-and-age-classification"
with zipfile.ZipFile("./adience-benchmark-gender-and-age-classification/"+Dataset+".zip", "r") as z:
    z.extractall(".")
