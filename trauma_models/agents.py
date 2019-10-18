import random
import uuid
import sys

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.UserParam import UserSettableParameter
from mesa.time import RandomActivation, BaseScheduler
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import ChartModule
from mesa.visualization.ModularVisualization import ModularServer

from adjustable_values import *
from place_agent import *


'''this will have all the relevant arguments to make each sub-class unique'''
class Agent_Class(Agent):
    grid = None
    x = None
    y = None
    moore = True

    """ An agent with fixed initial well_being."""
    def __init__(self, unique_id, pos, age, model, moore, well_being, total_trauma, total_success
        , radius_val,
        heritable_success, heritable_trauma, heritable_success_sensitivity,
        heritable_trauma_sensitivity, env_trauma, env_success, env_trauma_sensitivity,
        env_success_sensitivity, agent_heritable_success_proportion, agent_heritable_trauma_proportion,
        agent_env_success_proportion, agent_env_trauma_proportion,
        general_percent_change,
        standard_agent_radius,
        leader_search_agent_radius,
        trauma_event,
        age_dependency,
        model_max_age,
        leader_trauma_sensitivty_threshold,
        minimum_pregnancy_age,
        pregnancy_chance,
        leader_status, agent_type):

        super().__init__(self, model)
        self.unique_id = unique_id
        self.well_being = well_being
        self.heritable_success = heritable_success
        self.heritable_trauma = heritable_trauma
        self.heritable_success_sensitivity = heritable_success_sensitivity
        self.heritable_trauma_sensitivity = heritable_trauma_sensitivity
        self.env_trauma = env_trauma
        self.env_success = env_success
        self.env_trauma_sensitivity = env_trauma_sensitivity
        self.env_success_sensitivity = env_success_sensitivity
        self.pos = pos
        self.moore = moore
        self.age = age
        self.radius_val = radius_val
        self.leader_status = leader_status
        self.agent_type = agent_type
        self.total_trauma = total_trauma
        self.total_success = total_success
        self.agent_heritable_success_proportion = agent_heritable_success_proportion
        self.agent_heritable_trauma_proportion= agent_heritable_trauma_proportion
        self.agent_env_success_proportion=agent_env_success_proportion
        self.agent_env_trauma_proportion=agent_env_trauma_proportion
        self.general_percent_change = general_percent_change
        self.standard_agent_radius = standard_agent_radius
        self.leader_search_agent_radius = leader_search_agent_radius
        self.trauma_event = trauma_event
        self.age_dependency = age_dependency
        self.model_max_age = model_max_age
        self.leader_trauma_sensitivty_threshold = leader_trauma_sensitivty_threshold
        self.minimum_pregnancy_age = minimum_pregnancy_age
        self.pregnancy_chance = pregnancy_chance
        self.accum_env_trauma = []
        self.accum_env_success = []


    def move_similar_well_being(self):
        '''this will move in the area of a neighbour with similar well-being unless there isn't
        one then it remains'''
        neighbors = self.model.grid.get_neighbors(self.pos, self.moore,
            include_center=False, radius=self.radius_val)

        neighbors1 = [obj for obj in neighbors if isinstance(obj, (Agent_Class))]

        list_k = []
        list_v = []

        for i in neighbors1:
            list_k.append(i.well_being)
            list_v.append(i.pos)
        dict_well_being = dict(zip(list_k, list_v))
        '''if there are no neighbour agents (empty list_k) in radius, remain in position'''
        if len(list_k) < 1:
            new_pos = self.pos
        '''if there are neighbour agents'''
        if len(list_k) > 0:
            #this finds the similar value well-being by returning the one with the smallest difference
            similar_well_being = min(list_k, key=lambda x:abs(x-self.well_being))
            #then return the coords attached to that value
            ideal_pos = dict_well_being.get(similar_well_being)

            '''this produces a list of empty cells surrounding the ideal_pos, puts it into
            a sorted list and then returns the top of the list as the ideal coordinate'''

            ideal_empty_pos = [i for i in [obj for obj in
            (self.model.grid.get_neighborhood(ideal_pos,self.moore, include_center=False, radius=self.radius_val))] if self.model.grid.is_cell_empty(i)]
            ##print("*************** length of ideal pos", len(ideal_empty_pos), "\n empty pos", ideal_empty_pos)

            if len(ideal_empty_pos) <1:
                new_pos = self.pos
            else:
                new_pos = self.random.choice(ideal_empty_pos)

        new_position = self.model.grid.move_agent(self, new_pos)

    def ageing(self):

        self.age = self.age +1

    def leader_trans(self):

        '''leaders are decided by a chance lottery, only those with a low enough total trauma rate can win, currently average_success
        to 5-10%'''
        if self.leader_status is None:

            if self.env_trauma_sensitivity < self.leader_trauma_sensitivty_threshold:
                self.leader_status = 'yes'
                self.heritable_success = np.random.normal(leader_mean_heritable_success)
                self.env_success = np.random.normal(leader_mean_env_success)
                self.heritable_trauma = np.random.normal(leader_mean_heritable_trauma)
                self.env_trauma = np.random.normal(leader_mean_env_trauma)
            else:
                self.leader_status = 'no'

    def adjust_succ_sens(self):

        if self.env_success > success_inc_thres:

            Formulae.increase_succ_sens(self)

    def check_neighbors(self, ag_radius, ag_class):

        self.this_cell = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True, radius=self.standard_agent_radius)
        self.cellmates = [obj for obj in self.this_cell if isinstance(obj, (ag_class))]

    def rec_success(self):

        try:

            self.check_neighbors(self.leader_search_agent_radius, Agent_Class)
            leader_list = []
            if len(self.cellmates) > 0:
                for i in self.cellmates:
                    if i.leader_status is 'yes':
                        leader_list.append(i)
                        #print("leader tot success", i.total_success)
            if len(leader_list) > 0:
                other = random.choice(leader_list)

            try:
                self.env_success = round((other.total_success/self.env_success_sensitivity))
            except:
                self.env_success = 0

            self.accum_env_success.append(self.env_success)
            self.env_success = round(sum(self.accum_env_success)/(self.age))

        except:
            pass
        try:

            self.check_neighbors(self.standard_agent_radius, Agent_Class)
            if len(self.cellmates) > 0:
                other = random.choice(self.cellmates)

            try:
                self.env_success = round((other.total_success/self.env_success_sensitivity))

            except:

                self.env_success = 0
            self.accum_env_success.append(self.env_success)
            self.env_success = round(sum(self.accum_env_success)/(self.age))

        except:
            pass

    def rec_trauma(self):

        try:
            self.check_neighbors(self.leader_search_agent_radius, Agent_Class)
            leader_list = []
            if len(self.cellmates) > 0:
                for i in self.cellmates:
                    if i.leader_status is 'yes':
                        leader_list.append(i)

            if len(leader_list) > 0:
                other = random.choice(leader_list)

            try:
                #print("try")
                self.env_trauma = round((other.total_trauma/self.env_trauma_sensitivity))

            except:
                #print("except")
                self.env_trauma = 0

            self.accum_env_trauma.append(self.env_trauma)
            self.env_trauma = round(sum(self.accum_env_trauma)/(self.age))


        except:
            #print("final except")
            pass
        try:
            #print("try reg starteeee")
            #print("standard_ag_rad", self.standard_agent_radius)
            self.check_neighbors(self.standard_agent_radius, Agent_Class)
            #print("len cellmates", len(self.cellmates))
            if len(self.cellmates) > 0:
                other = random.choice(self.cellmates)

            #try:
                #print("try inner reg")
                #print("env_trauma", self.env_trauma)
                self.env_trauma = round((other.total_trauma/self.env_trauma_sensitivity))
        except:
            #print("except reg")
            self.env_trauma = 0



        else:
            #print("final except reg")
            pass
        self.accum_env_trauma.append(self.env_trauma)
        self.env_trauma = round(sum(self.accum_env_trauma)/(self.age))


    def calc_total_t_s(self):

        self.total_trauma = round(((self.env_trauma/100)*self.agent_env_trauma_proportion +
        (self.heritable_trauma/100)*self.agent_heritable_trauma_proportion))


    

        self.total_success = round(((self.env_success/100)*self.agent_env_success_proportion +
        (self.heritable_success/100)*self.agent_heritable_success_proportion))

    def remove_this_agent(self):
        self.model.grid._remove_agent(self.pos, self)

        self.model.schedule.remove(self)

    def death_age(self):

        if self.age >= self.model_max_age:

            try:
                self.remove_this_agent()
            except:
                pass

    def calculate_well_being(self):
        self.well_being = self.total_success - self.total_trauma

    def cap_max_vals(self):

        if self.env_trauma > 10:
            self.env_trauma = 10
        if self.env_success > 10:
            self.env_success = 10
        if self.env_success < 1:
            self.env_success = 1


    def death_trauma(self):
        if self.total_trauma > max_trauma:

            try:
                self.remove_this_agent()
            except:
                pass

    def death_success(self):
        if self.total_success > min_success:
            try:
                self.remove_this_agent()
            except:
                pass
    def sens_adjust(self):
        #this stops the model from crashing if values get below zero, agents should be able to have sensitivty below 0 anyway
        if self.env_success_sensitivity < 0.51:
            self.env_success_sensitivity = 0.6
        if self.heritable_success_sensitivity < 0.51:
            self.heritable_success_sensitivity = 0.6

        if self.env_trauma_sensitivity < 0.51:
            self.env_trauma_sensitivity = 0.6
        if self.heritable_trauma_sensitivity < 0.51:
            self.heritable_trauma_sensitivity = 0.6

    def no_env_vertical_only(self):
        self.env_trauma = 0
        self.env_trauma_sensitivity = 0
        self.env_success = 0
        self.env_success_sensitivity = 0

    def age_dep_ptsd(self):
        '''this will change the heritable trauma and success value to a ptsd range for a short period after the onset of the trauma
        in accordance with research indicating 0-2 yrs being the age group that leads to impaired cognitive and long term
        genetic changes'''

        if self.model.schedule.time <= self.trauma_event:

            if random.randint(0,100) < perc_after_exposure:

                self.heritable_success = round(np.random.normal(heritable_ptsd_suc_rate))
                self.heritable_trauma = round(np.random.normal(heritable_ptsd_trauma_rate))


    def reproduction(self):

        if self.age > self.minimum_pregnancy_age:

            if random.randint(0,100) < self.pregnancy_chance:

                #trauma sens goes down by 6% and success up by 6% each new birth, to signify the decreasing potency
                #as we move away from the traumatic event 300 yrs/the max of the scale (50)
                Upgrade_Agent.place_off_spring(self, agent_class=Agent_Class, pos=self.pos, init_age=0,
                #normal distributed heritable vals
                mean_parent_h_trauma= round(self.heritable_trauma), #round((self.heritable_trauma-((self.heritable_trauma/100)*percent_change))),
                mean_parent_h_success= round(self.heritable_success), #round((self.heritable_success-((self.heritable_success/100)*percent_change))),
                mean_parent_h_succ_sens= round(self.heritable_success_sensitivity), #((self.heritable_success_sensitivity-((self.heritable_success_sensitivity/100)*percent_change))),
                mean_parent_h_trau_sens= round(self.heritable_trauma_sensitivity), #round(self.heritable_trauma_sensitivity-((self.heritable_trauma_sensitivity/100)*percent_change)),
                #the sensitivty to the env transmission goes down as the environment moves away from primary

                mean_parent_env_succ_sens= round(self.env_success_sensitivity + ((self.env_success_sensitivity/100)*self.general_percent_change)),
                mean_parent_env_trau_sens= round(self.env_trauma_sensitivity-((self.env_trauma_sensitivity/100)*self.general_percent_change)),
                #env starts a fresh
                mean_parent_env_trauma= 0, #round(self.env_trauma-((self.env_trauma/100)*percent_change)),
                mean_parent_env_success= 0, #round(self.env_success + ((self.env_success/100)*percent_change)),
                success_heritable_proportion = self.agent_heritable_success_proportion,
                trauma_heritable_proportion = self.agent_heritable_trauma_proportion,
                success_env_proportion = self.agent_env_success_proportion,
                trauma_env_proportion = self.agent_env_trauma_proportion,
                general_percent_change =  self.general_percent_change,
                standard_agent_radius =  self.standard_agent_radius,
                leader_search_agent_radius =  self.leader_search_agent_radius,
                trauma_event =  self.trauma_event,
                age_dependency =  self.age_dependency,
                model_max_age =  self.model_max_age,
                leader_trauma_sensitivty_threshold =  self.leader_trauma_sensitivty_threshold,
                minimum_pregnancy_age =  self.minimum_pregnancy_age,
                pregnancy_chance =  self.pregnancy_chance
                )




    def step(self):



        if analysis_by_exp is 'vertical_age_dpt_leaders':

            self.ageing()
            self.age_dep_ptsd()
            self.leader_trans()
            self.move_similar_well_being()
            self.rec_success()
            self.rec_trauma()
            self.cap_max_vals()
            self.calc_total_t_s()
            self.calculate_well_being()

            self.reproduction()
            self.sens_adjust()
            self.death_trauma()
            self.death_age()

        if analysis_by_exp is 'vertical_age_dpt_horizontal':

            self.ageing()
            self.age_dep_ptsd()
            self.move_similar_well_being()
            self.rec_success()
            self.rec_trauma()
            self.cap_max_vals()
            self.calc_total_t_s()
            self.calculate_well_being()
            self.death_trauma()
            self.death_age()
            self.reproduction()
            self.sens_adjust()

        if analysis_by_exp is 'vertical_age_dpt':
            self.no_env_vertical_only()
            self.ageing()
            self.age_dep_ptsd()
            self.move_similar_well_being()
            self.cap_max_vals()
            self.calc_total_t_s()
            self.calculate_well_being()
            self.reproduction()
            self.sens_adjust()
            self.death_trauma()
            self.death_age()


        if analysis_by_exp is 'vertical_only':
            self.no_env_vertical_only()
            self.ageing()
            self.move_similar_well_being()
            self.cap_max_vals()
            self.calc_total_t_s()
            self.calculate_well_being()
            self.reproduction()
            self.sens_adjust()
            self.death_trauma()
            self.death_age()
