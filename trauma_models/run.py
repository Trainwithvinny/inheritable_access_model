from server import server
'''something from server doesn't work on small laptop'''
from model import MoneyModel
from data_vis_mpl import Data_Visualisation
from adjustable_values import *


while True:
    # some code here
    print("Have you double checked the run.py page? batchrunner \n will take a while to re-run and you will lose saved previous runs")
    if input('Do You Want To Continue? y or n \n') != 'n':
        break
server.launch()
#MoneyModel().run_model()
MoneyModel().batchrunner()
#Data_Visualisation().run_exploratory_analysis()
#Data_Visualisation().run_analytics()
