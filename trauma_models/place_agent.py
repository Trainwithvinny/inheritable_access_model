import random
from mesa import Model
import uuid
from mesa.space import MultiGrid
import uuid
from adjustable_values import *
import numpy as np

'''this class deals with each new agent and their inheritance'''
class Upgrade_Agent(Model):
    def __init__(self):
        #self.agent_class = agent_class
        pass

    def place_off_spring(self,
    agent_class,
    pos,
    init_age,
    mean_parent_h_succ_sens,
    mean_parent_h_trau_sens,
    mean_parent_env_succ_sens,
    mean_parent_env_trau_sens,
    mean_parent_env_trauma,
    mean_parent_env_success,
    mean_parent_h_success,
    mean_parent_h_trauma,
    success_heritable_proportion,
    trauma_heritable_proportion,
    success_env_proportion,
    trauma_env_proportion,
    general_percent_change,
    standard_agent_radius,
    leader_search_agent_radius,
    trauma_event,
    age_dependency,
    model_max_age,
    leader_trauma_sensitivty_threshold,
    minimum_pregnancy_age,
    pregnancy_chance,
    ):
        #grid_width = 20
        #grid_height = 20


        #self.unique_id = uuid.uuid4()
        self.agent_heritable_success_proportion = success_heritable_proportion
        self.agent_type = 'agent'
        self.radius_val = off_spring_radius
        self.age=init_age

        self.agent_heritable_success_proportion = success_heritable_proportion
        x = self.random.randrange(grid_width)
        y = self.random.randrange(grid_height)

        self.heritable_success = round(np.random.normal(mean_parent_h_success, heri_std_scale_norm_dist)) #heri says how flat or belled
        self.heritable_trauma = round(np.random.normal(mean_parent_h_trauma, heri_std_scale_norm_dist))
        self.well_being = self.total_success - self.total_trauma
        self.heritable_success_sensitivity = round(np.random.normal(mean_parent_h_succ_sens, heri_std_scale_norm_dist))
        self.heritable_trauma_sensitivity = round(np.random.normal(mean_parent_h_trau_sens, heri_std_scale_norm_dist))
        self.env_trauma_sensitivity = round((np.random.normal(self.heritable_trauma)))#/max_heritable_trau_val
        self.env_success_sensitivity = round((np.random.normal(self.heritable_success)))#/max_heritable_succ_val
        #print("env trauma sens", self.env_trauma_sensitivity)
        self.env_trauma = mean_parent_env_trauma
        self.env_success = mean_parent_env_success
        self.total_trauma = self.env_trauma + self.heritable_trauma
        self.total_success = self.env_success + self.heritable_success
        #self.success_trauma = self.random.randrange(10, 1)
        self.well_being = self.total_success - self.total_trauma
        #self.trauma_event = trauma_event

        #agents are born by their parents so environmental factors are similar
        agent = agent_class(unique_id= self.model.next_id(), pos= (x,y), age= self.age, model= self.model,
            moore= True, well_being= self.well_being, agent_type=self.agent_type, total_trauma= self.total_trauma, total_success= self.total_success ,
            radius_val= self.radius_val,
            heritable_success= self.heritable_success, heritable_trauma= self.heritable_trauma,
            heritable_success_sensitivity= self.heritable_success_sensitivity,
            heritable_trauma_sensitivity= self.heritable_trauma_sensitivity, env_trauma= self.env_trauma, env_success= self.env_success,
            env_trauma_sensitivity= self.env_trauma_sensitivity,

            env_success_sensitivity= self.env_success_sensitivity,
            #the below syntax can be confusing so pay close attention to the name as it changes
            #between classes to help know which belongs where
            agent_heritable_success_proportion = self.agent_heritable_success_proportion,
            agent_heritable_trauma_proportion = self.agent_heritable_trauma_proportion,
            agent_env_success_proportion = self.agent_env_success_proportion,
            agent_env_trauma_proportion = self.agent_env_trauma_proportion,
            general_percent_change = self.general_percent_change,
            standard_agent_radius = self.standard_agent_radius,
            leader_search_agent_radius = self.leader_search_agent_radius,
            trauma_event = self.trauma_event,
            age_dependency = self.age_dependency,
            model_max_age = self.model_max_age,
            leader_trauma_sensitivty_threshold = self.leader_trauma_sensitivty_threshold,
            minimum_pregnancy_age = self.minimum_pregnancy_age,
            pregnancy_chance = self.pregnancy_chance,
            #leader_perc=self.leader_perc,
            leader_status=None)


        self.model.grid.place_agent(agent, pos)
        #print("off spring id", self.unique_id)
        #RandomActivation.add(agent)
        self.model.schedule.add(agent)
