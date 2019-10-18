import random
import uuid
from collections import OrderedDict

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from agents import *
from adjustable_values import *
from schedule import RandomActivationByType
from mesa.batchrunner import BatchRunner
import pandas as pd
import numpy as np
import os
from place_agent import *


class MoneyModel(Model):
    """A model with some number of agents."""
    height = grid_height
    width = grid_width

    Verbose = False # if verbose include print statements


    description = 'First model observing the effects of trauma'

    def __init__(self,
                initial_num = initial_num_val,
                mean_heritable_success = mean_heritable_success_val,
                mean_heritable_trauma = mean_heritable_trauma_val,
                mean_env_trauma = mean_env_trauma_val,
                mean_env_success = mean_env_success_val,
                mean_hertiable_suc_sens = mean_hertiable_suc_sens_val,
                mean_heritable_trau_sens = mean_heritable_trau_sens_val,
                mean_env_succ_sens = mean_env_succ_sens_val,
                mean_env_trau_sens = mean_env_trau_sens_val,
                width = grid_width,
                height = grid_height,
                success_heritable_proportion= success_heritable_proportion_val,
                success_env_proportion= success_env_proportion_val,
                trauma_heritable_proportion= trauma_heritable_proportion_val,
                trauma_env_proportion= trauma_env_proportion_val,
                general_percent_change= percent_change_val,
                standard_agent_radius= standard_ag_rad_val,
                leader_search_agent_radius= leader_search_ag_rad_val,
                trauma_event= trauma_event_fp_duration_val,
                age_dependency= set_exp_age_val,
                model_max_age= max_age_val,
                leader_trauma_sensitivty_threshold= leader_e_trau_sens_thres_val,
                minimum_pregnancy_age= preg_min_age_val,
                pregnancy_chance= preg_perc_int_val,
                runs_model=0

                ):

        super().__init__()
        self.my_path = os.path.abspath(__file__)
        self.mother_folder, data_vis_py = os.path.split(self.my_path)
        self.grouped_folder, models = os.path.split(self.mother_folder)
        self.sensitivty_csv_folder = os.path.join(self.grouped_folder, "sensitivity_analysis", "csv_folder")
        self.model_output_folder = os.path.join(self.grouped_folder, "model_output")
        self.data_folder = os.path.join(self.model_output_folder, analysis_by_exp)
        self.csv_folder = os.path.join(self.data_folder, "csv_folder")
        self.filename1 = "step_data_batch_hoz_" + analysis_by_exp + ".csv"
        self.filename2 = "data_agent_" + analysis_by_exp + ".csv"
        self.filename3 = "br_df_model_vars.csv"
        self.filename4 = "i_run_data.csv"
        self.filename5 = "line3"
        self.filename6 = "full_sa_step_data_batch_hoz_" + analysis_by_exp + ".csv"

        self.width = grid_width
        self.height = grid_height
        self.initial_num = initial_num
        self.grid = MultiGrid(self.width, self.height, torus=True)
        self.schedule = RandomActivationByType(self)
        self.br_df = None
        self.br_step_data = None
        self.whole_df = None

        #br_params
        self.success_heritable_proportion=success_heritable_proportion
        self.success_env_proportion=success_env_proportion
        self.trauma_heritable_proportion=trauma_heritable_proportion
        self.trauma_env_proportion=trauma_env_proportion

        #self.init_initial_num= 30
        self.mean_heritable_success = mean_heritable_success
        self.mean_heritable_trauma = mean_heritable_trauma
        self.mean_env_trauma = mean_env_trauma
        self.mean_env_success = mean_env_success
        self.mean_heritable_trau_sens = mean_heritable_trau_sens
        self.mean_hertiable_suc_sens = mean_hertiable_suc_sens
        self.mean_env_succ_sens = mean_env_succ_sens
        self.mean_env_trau_sens = mean_env_trau_sens
        self.heritable_ptsd_trauma_rate = heritable_ptsd_trauma_rate
        self.heritable_ptsd_suc_rate = heritable_ptsd_suc_rate
        self.leader_e_trau_sens_thres = leader_e_trau_sens_thres
        self.general_percent_change = general_percent_change
        self.death_percent_model = 0
        self.population_growth_rate = 0
        self.trauma_event = trauma_event
        #self.general_percent_change = general_percent_change
        #single run params
        self.success_heritable_proportion= success_heritable_proportion
        self.success_env_proportion= success_env_proportion
        self.trauma_heritable_proportion= trauma_heritable_proportion
        self.trauma_env_proportion= trauma_env_proportion
        self.general_percent_change= general_percent_change
        self.standard_agent_radius= standard_agent_radius
        self.leader_search_agent_radius= leader_search_agent_radius
        self.trauma_event= trauma_event
        self.age_dependency= set_exp_age_val
        self.model_max_age= max_age_val
        self.leader_trauma_sensitivty_threshold= leader_trauma_sensitivty_threshold
        self.minimum_pregnancy_age= minimum_pregnancy_age
        self.pregnancy_chance= pregnancy_chance
        #self.runs_model = runs_model
        self.runs_model = runs_model_var
        self.datacollector = DataCollector(model_reporters=
            {#"run_number": lambda m: m.run_number
            "initial_num": lambda m: m.schedule.get_initial_num(MoneyModel),
            "mean_heritable_success": lambda m: m.schedule.get_mean_heritable_succes(MoneyModel),
            "mean_heritable_trauma": lambda m: m.schedule.get_mean_heritable_trauma(MoneyModel),
            "mean_env_trauma": lambda m: m.schedule.get_mean_env_trauma(MoneyModel),
            "mean_env_success": lambda m: m.schedule.get_mean_env_success(MoneyModel),
            "mean_heritable_trau_sens": lambda m: m.schedule.get_mean_heritable_trau_sens(MoneyModel),
            "mean_hertiable_suc_sens": lambda m: m.schedule.get_mean_hertiable_suc_sens(MoneyModel),
            "mean_env_succ_sens": lambda m: m.schedule.get_mean_env_succ_sens(MoneyModel),
            "mean_env_trau_sens": lambda m: m.schedule.get_mean_env_trau_sens(MoneyModel),
            "model_tick": lambda m: m.schedule.time,
            "agent_avg_age": lambda m: m.schedule.agent_avg_age(Agent_Class),
            "trauma_heritable_proportion": lambda m: m.schedule.get_trauma_heritable_proportion(MoneyModel),
            "trauma_env_proportion": lambda m: m.schedule.get_trauma_env_proportion(MoneyModel),
            "success_heritable_proportion": lambda m: m.schedule.get_success_heritable_proportion(MoneyModel),
            "success_env_proportion": lambda m: m.schedule.get_success_env_proportion(MoneyModel),
            "average_well_being": lambda m: m.schedule.get_agents_well_being(Agent_Class),
            "average_env_trauma": lambda m: m.schedule.get_agents_env_trauma(Agent_Class, attribute="env_trauma"),
            "average_h_trauma": lambda m: m.schedule.get_agents_h_trauma(Agent_Class, attribute="h_trauma"),
            "average_total_trauma": lambda m: m.schedule.get_agents_trauma(Agent_Class),
            "average_env_success": lambda m: m.schedule.get_agents_env_success(Agent_Class, attribute="env_success"),
            "average_h_success": lambda m: m.schedule.get_agents_h_success(Agent_Class, attribute="h_success"),
            "average_total_success": lambda m: m.schedule.get_agents_success(Agent_Class),
            "ptsd_percent": lambda m: m.schedule.ptsd_percentage(Agent_Class),
            "leader_count": lambda m: m.schedule.leaders_count(Agent_Class),
            "leader_perc": lambda m: m.schedule.leaders_perc(Agent_Class),
            #"percent_change": lambda m: m.perc_change,
            "leader_avg_trauma": lambda m: m.schedule.get_leaders_tot_trauma(Agent_Class),
            "leader_avg_success": lambda m: m.schedule.get_leaders_tot_success(Agent_Class),
            "agent_by_age_2": lambda m: m.schedule.agent_by_age(0, 2, Agent_Class),
            "percent_change": lambda m: m.general_percent_change,
            "agent_radius": lambda m: m.standard_agent_radius,
            "leader_search_ag_rad": lambda m: m.leader_search_agent_radius,
            "event_duration": lambda m: m.trauma_event,
            "set_exp_age": lambda m: m.age_dependency,
            "max_age": lambda m: m.model_max_age,
            "leader_trau_thres": lambda m: m.leader_trauma_sensitivty_threshold,
            "min_preg_age": lambda m: m.minimum_pregnancy_age,
            "preg_chance": lambda m: m.pregnancy_chance,
            #"death_rate": lambda m: m.schedule.get_death_rate(MoneyModel),
            "death_percent": lambda m: m.death_percent_model,
            "pop_growth_rate": lambda m: m.population_growth_rate,
            "agent_count": lambda m: m.schedule.get_type_count(Agent_Class)})

        self.ar = {"agent_age": lambda a: a.age,
        "agent_well_being": lambda a: a.well_being,
        "leader_status": lambda a: a.leader_status,
        "total_trauma": lambda a: a.total_trauma,
        "total_success": lambda a: a.total_success,
        "env_success": lambda a: a.env_success, "env_trauma": lambda a: a.env_trauma,
        "env_trauma_sens": lambda a: a.env_trauma_sensitivity, "env_success_sens": lambda a: a.env_success_sensitivity,
        "h_trauma": lambda a: a.heritable_trauma, "h_success": lambda a: a.heritable_success,
        "h_trauma_sens": lambda a: a.heritable_trauma_sensitivity, "h_success_sens": lambda a: a.heritable_success_sensitivity,
        "radius": lambda a: a.radius_val}
        self.dc = DataCollector(agent_reporters=self.ar)


        def place_agent_start(agent_class):


            print("self.initial_num", (self.initial_num))
            for i in range(self.initial_num):

                self.agent_type = 'agent'
                '''stick with normal distribution for conitnuinty and ease also the greater the sample the size
                the more chance for outliers and variability'''

                self.radius_val = initial_agent_radius
                
                self.age=round(np.random.normal(mean_age))

                x = self.random.randrange(compact_grid_width)
                y = self.random.randrange(compact_grid_height)
                if analysis_by_exp is not 'vertical_only':
                    if self.age > self.age_dependency:
                        self.heritable_success = round(np.random.normal(self.mean_heritable_success))
                        self.heritable_trauma = round(np.random.normal(self.mean_heritable_trauma))
                    else:
                        self.heritable_success = round(np.random.normal(self.heritable_ptsd_suc_rate))
                        self.heritable_trauma = round(np.random.normal(self.heritable_ptsd_trauma_rate))
                if analysis_by_exp is 'vertical_only':
                    self.heritable_success = round(np.random.normal(self.mean_heritable_success))
                    self.heritable_trauma = round(np.random.normal(self.mean_heritable_trauma))
                self.env_success = round(np.random.normal(self.mean_env_success))
                self.env_trauma = round(np.random.normal(self.mean_env_trauma))

                '''linking environmental sensitivty to heritable values then using as a percentage'''

                self.env_trauma_sensitivity = round((np.random.normal(self.heritable_trauma)))#/max_heritable_trau_val

                self.env_success_sensitivity = round((np.random.normal(self.heritable_success)))#/max_heritable_succ_val

                self.heritable_success_sensitivity = round(np.random.normal(self.mean_hertiable_suc_sens))

                #self.random.uniform(min_suc_sens max_suc_sens)
                self.heritable_trauma_sensitivity = round(np.random.normal(self.mean_heritable_trau_sens))


                #do I need these totals as its done in the steps
                self.total_success = round((((self.env_success/100)*self.success_env_proportion) + ((self.heritable_success/100)*self.success_heritable_proportion)))
                self.total_trauma = round(((self.env_trauma/100)*self.trauma_env_proportion) +  ((self.heritable_trauma/100)*self.trauma_heritable_proportion))
                self.well_being = round(self.total_success - self.total_trauma)

                agent = agent_class(unique_id= self.next_id(), pos= (x,y), age= self.age, model= self, moore= True,
                    well_being= self.well_being, agent_type='agent', total_trauma=self.total_trauma, total_success=self.total_success,
                     radius_val= self.radius_val,
                    heritable_success= self.heritable_success, heritable_trauma= self.heritable_trauma,
                    heritable_success_sensitivity= self.heritable_success_sensitivity,
                    heritable_trauma_sensitivity= self.heritable_trauma_sensitivity, env_trauma= self.env_trauma,
                    env_success= self.env_success, env_trauma_sensitivity= self.env_trauma_sensitivity,
                    env_success_sensitivity= self.env_success_sensitivity,
                    agent_heritable_success_proportion=self.success_heritable_proportion, agent_heritable_trauma_proportion=self.trauma_heritable_proportion,
                    agent_env_success_proportion=self.success_env_proportion, agent_env_trauma_proportion=self.trauma_env_proportion,
                    general_percent_change = self.general_percent_change,
                    standard_agent_radius = self.standard_agent_radius,
                    leader_search_agent_radius = self.leader_search_agent_radius,
                    trauma_event = self.trauma_event,
                    age_dependency = self.age_dependency,
                    model_max_age = self.model_max_age,
                    leader_trauma_sensitivty_threshold = self.leader_trauma_sensitivty_threshold,
                    minimum_pregnancy_age = self.minimum_pregnancy_age,
                    pregnancy_chance = self.pregnancy_chance,
                     leader_status=None)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)


        place_agent_start(agent_class=Agent_Class)

        self.running = True

    def run_model(self):
        '''this will produce a certain amount of runs (100) for one set of variables'''
        for i in range(1):
            for i in range(runs_model_var):
                #print("model tick", i)
                self.step()

        print("Doing agent table to: ", self.csv_folder, "name: ", self.filename2)
        br_agent_df = self.dc.get_agent_vars_dataframe()
        br_agent_df.to_csv(os.path.join(self.csv_folder, self.filename2))

    def batchrunner(self):
        # "leader_perc":leader_perc_br
        br_params = {
        "general_percent_change": percent_change,
        "standard_agent_radius": standard_ag_rad,
        "leader_search_agent_radius": leader_search_ag_rad,
        #"trauma_event": trauma_event_fp_duration,
        "age_dependency": set_exp_age,
        "model_max_age": max_age,
        "leader_trauma_sensitivty_threshold": leader_e_trau_sens_thres,
        "minimum_pregnancy_age": preg_min_age,
        "general_percent_change": percent_change,
        "standard_agent_radius": standard_ag_rad,
        "trauma_event": trauma_event_fp_duration,
        "pregnancy_chance": preg_perc_int,
        "initial_num": init_initial_num
        } #this can do different number of runs per batch

        br = BatchRunner(model_cls=MoneyModel,
                         variable_parameters=br_params,
                         fixed_parameters=None,#fixed for each batch run
                         iterations=model_iterations,
                         max_steps=model_max_steps,
                         model_reporters={"Data Collector": lambda m: m.datacollector},
                        #agent_reporters={"Agent Collector": lambda a: a.dc},
                         display_progress=True
                         )


        br.run_all()
        br_df = br.get_model_vars_dataframe()
        br_step_data = pd.DataFrame()

        for i in range(len(br_df["Data Collector"])):
            if isinstance(br_df["Data Collector"][i], DataCollector):
                i_run_data = br_df["Data Collector"][i].get_model_vars_dataframe()
                br_step_data = br_step_data.append(i_run_data, ignore_index=True)

        br_step_data.to_csv(os.path.join(self.csv_folder, self.filename1)) #inheritableaccessmodel_step_data_batch_hoz

        self.br_step_data = br_step_data
        return br_step_data

    def agent_df(self):

        br_agent_df = self.dc.get_agent_vars_dataframe()
        #print("agent df" br_agent_df)
        br_agent_df.to_csv(os.path.join(self.csv_folder, self.filename2))

    def step(self):
        total_beginning = self.schedule.get_type_count(Agent_Class)
        #print("agent count", total_beginning)
        self.datacollector.collect(self) #this is required to collect the data per step
        self.dc.collect(self) #ditto
        self.schedule.get_agents_well_being(Agent_Class)
        self.schedule.get_agents_success(Agent_Class)
        self.schedule.get_agents_trauma(Agent_Class)
        self.schedule.step()
        self.schedule.death_percent_rate(total_beginning)
        self.schedule.population_growth_rate(total_beginning)
