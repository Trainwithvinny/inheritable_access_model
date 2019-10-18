from server import server
'''something from server doesn't work on small laptop'''
from model import MoneyModel
from data_vis_mpl import Data_Visualisation
from adjustable_values import *



server.launch()
#MoneyModel().run_model()
MoneyModel().batchrunner()
#Data_Visualisation().run_exploratory_analysis()
#Data_Visualisation().run_analytics()
