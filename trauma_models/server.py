import random
import uuid

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.UserParam import UserSettableParameter


from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import ChartModule
from mesa.visualization.ModularVisualization import ModularServer

from model import MoneyModel
from agents import Agent_Class #, Leader_Agent, Off_Spring_Agent
#from resource_patches import *
from adjustable_values import *



def agent_portrayal(agent):


    if agent is None:
        return

    portrayal = {}

    if type(agent) is Agent_Class:

        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "r": 0.5}
        if agent.age >= 16:

            portrayal["Color"] = "yellow"
            portrayal["Layer"] = 4
            portrayal["r"] = 0.4
            if agent.leader_status is 'yes':
                portrayal["Color"] = "yellow"
                portrayal["Layer"] = 5
                portrayal["r"] = 0.5

        if agent.age < 16:

            portrayal["Color"] = "brown"
            portrayal["Layer"] = 2
            portrayal["r"] = 0.2
            if agent.leader_status is 'yes':
                portrayal["Color"] = "brown"
                portrayal["Layer"] = 3
                portrayal["r"] = 0.3



    return portrayal



#CanvasGrid is creating the portrayal for the json
grid = CanvasGrid(agent_portrayal, 100, 100, 1000, 1000)
'''it is possible to have the agent activity mapped onto a visual in the browser'''
'''chart = ChartModule([
    {"Label": "Gini", "Color": "Black"}],
    data_collector_name='datacollector'
)'''
'''chart_element = ChartModule([{"Label": "Agent", "Color": "#AA0000"},
                             {"Label": "Average Well-Being", "Color": "#666666"}])
'''
model_params = {#"grass": UserSettableParameter('checkbox', 'Grass Enabled', True),
                "initial_num": UserSettableParameter('slider', 'Initial number of agents', 10, 1, 100),
                "runs_model": UserSettableParameter('slider', 'Iterations', 10, 1, 100),
                #"initial_parents": UserSettableParameter('slider', 'Initial parents quantity', 20, 1, 100),
                #"initial_leaders": UserSettableParameter('slider', 'Initial leaders quantity', 3, 1, 100),
                #"initial_offspring": UserSettableParameter('slider', 'Initial off-spring quantity', 10, 1, 100),
                "initial_success_store": UserSettableParameter('slider', 'Quantity of success patches', 10, 1, 100),
                #"leader_well_being_thres": UserSettableParameter('slider', 'Well-being threshold for becoming Leader', 10, 1, 100),
                "initial_trauma_store": UserSettableParameter('slider', 'Quantity of trauma patches', 10, 1, 100),
                "success_quantity": UserSettableParameter('slider', 'Success stored in each patch', 10, 1, 100),
                "trauma_quantity": UserSettableParameter('slider', 'Trauma stored in each patch', 10, 1, 100),
                "success_intensity": UserSettableParameter('slider', 'Success patch transmission intensity', 10, 1, 100),
                "trauma_intensity": UserSettableParameter('slider', 'Trauma patch transmission intensity', 10, 1, 100),
                "success_sensitivity": UserSettableParameter('slider', 'Agent sensitivity to success', 10, 1, 100),
                "trauma_sensitivity": UserSettableParameter('slider', 'Agent sensitivity to trauma', 10, 1, 100),
                "time_in_steps_degrade": UserSettableParameter('slider', 'Time for patch to disappear', 10, 1, 100)
                }

server = ModularServer(MoneyModel, [grid], "Access Model")#[grid, chart_element] , model_params

server.port = 8889
