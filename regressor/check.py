import pandas as pd
from utils import handle_data
data = pd.read_csv("data/train.csv")
print(data.shape)

test = pd.read_csv("data/test.csv")
print(test.shape)
hd = handle_data(data = data)
num, cat, cont, disc, yr = hd.extract_var() 

print("Numerical : "+str(len(num))+", Categorical "+
      str(len(cat)) + ", Continuous: " + str(len(cont))+ ", Discrete: " + str(len(disc)))