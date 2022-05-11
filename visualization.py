import plotly.express as px
import numpy
from numpy import loadtxt
import pandas as pd
# Random Data
random_x = loadtxt('istanbul.txt', dtype='int')

a = pd.value_counts(random_x).plot.bar()
