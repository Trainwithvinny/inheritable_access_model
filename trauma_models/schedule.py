from collections import defaultdict
from agents import *
from mesa import Model
from collections import OrderedDict
from mesa.time import RandomActivation, BaseScheduler
import random
import numpy as np
from adjustable_values import *




class RandomActivationByType(RandomActivation):
    '''
    A scheduler which activates each type of agent once per step, in random
    order, with the order reshuffled every step.
    This is equivalent to the NetLogo 'ask type...' and is generally the
    default behavior for an ABM.
    Assumes that all agents have a step() method.
    '''

    def __init__(self, model):
        super().__init__(model)
        self.agents_by_type = defaultdict(dict)
        self._agents = defaultdict(dict)
        self.agents = defaultdict(dict)


    def add(self, agent):
        '''
        Add an Agent object to the schedule
        Args:
            agent: An Agent to be added to the schedule.
        '''
        #at the beginning of the run it will add agents
        #_agents is an ordereddict dict
        self._agents[agent.unique_id] = agent
        if hasattr(agent, 'agent_type'):
            self.agents[agent.unique_id] = agent #self.agents_by_type[class_type].values()

        ##print("_agents", agent) #this #prints the object

        agent_class = type(agent)
        ##print(",,,,,,,,,,,,,,,agents with class,,,,,,,,,,,,,", agent_class)
        #this outputs agents.Agent_Class & resource_patches.Well_Being_Patch
        self.agents_by_type[agent_class][agent.unique_id] = agent

        #print(",,,,,,,,,,,,,,,,agents and id???,,,,,,,,,,,,,", agent)
        #self.agent_class = agent_class
    def remove(self, agent):
        '''
        Remove all instances of a given agent from the schedule.
        '''
        #print("BEFORE del agents by type ordered dict", "\n ordered dict length", len(self._agents))
        #print("to be deleted", self._agents[agent.unique_id])
        del self._agents[agent.unique_id]
        del self.agents[agent.unique_id]

        agent_class = type(agent)
        #auid = self._agents[agent.unique_id]
        #print("to be deleted by type ", self.agents_by_type[agent_class][agent.unique_id])
        del self.agents_by_type[agent_class][agent.unique_id]
        #self.model.schedule.remove(self)
        #print("AFTER del agents by type ordered dict", "\n ordered dict length", len(self._agents)) #, "\n", self._agents)

    def step(self, by_type=True):
        '''
        Executes the step of each agent type, one at a time, in random order.
        Args:
            by_type: If True, run all agents of a single type before running
                      the next one.
        '''

        if by_type:
            for agent_class in self.agents_by_type:
                self.get_type_count(agent_class)
                ##print("agent class is ", agent_class)
                #this runs one class before the other
                self.step_type(agent_class) # change this to auid

            self.steps += 1
            self.time += 1
        else:
            super().step()

    def step_type(self, type):
        '''
        Shuffle order and run all agents of a given type.
        Args:
            type: Class object of the type to run.
        '''
        #this doesnt seem to shuffled, it gives an ordered list, as does wolf so its ok
        agent_keys = list(self.agents_by_type[type].keys())
        #print("****************AGENT KEYS***************", agent_keys)
        random.shuffle(agent_keys)
        #self.model.random.shuffle(agent_keys)
        #print("-------------------agents keys after shuffle ***********", agent_keys)

        for agent_key in agent_keys:
            self.agents_by_type[type][agent_key].step()

        

        '''THE REST OF THIS COLLECTS DIFFERENT DATA POINTS FOR THE MODEL AND AGENTS'''

    def get_type_count(self, type_class):
        '''
        Returns the current number of agents of certain type in the queue.
        '''
        #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx length of queue", len(self.agents_by_type[type_class].values()))
        #print("get agents", self.agents_by_type[Agent_Class])
        return len(self.agents_by_type[type_class].values())

    '''def get_current_agent(self, type_class):

        Returns the current number of agents of certain type in the queue.

        #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx length of queue", len(self.agents_by_type[type_class].values()))
        #print("get agents", self.agents_by_type[Agent_Class])
        try:
            return (self.agents_by_type[type_class].values())
        except:
            return(0)'''

    def death_rate(self, class_type, model):
        '''this kills agents at the provided death rate'''

        agents = list(self.agents_by_type[class_type].values())
        #print("agents", (type(agents)))
        #print("death_rate", type(death_rate))

        selection_num = int(np.round((len(agents)/100)*self.model.death_rate))
        #print("selection_num", selection_num)
        death_list = random.sample(agents, k=selection_num)
        for i in death_list:
            self.remove(i)
        #print(len(agents))
        #print(len(death_list))

    def death_percent_rate(self, total_beginning_value):

        total_end = self.get_type_count(Agent_Class)
        dead_agents = total_beginning_value - total_end
        death_percent_model = (dead_agents/total_beginning_value)*100
        self.model.death_percent_model = death_percent_model


    def population_growth_rate(self, total_beginning_value):
        total_end = self.get_type_count(Agent_Class)
        population_rate = ((total_end - total_beginning_value)/total_beginning_value)*100
        self.model.population_growth_rate = population_rate
        #print("growth", self.population_rate)


    def get_run_number(self):
        '''
        Returns the current number of agents of certain type in the queue.
        '''
        #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx length of queue", len(self.agents_by_type[type_class].values()))
        #print("get agents", self.agents_by_type[Agent_Class])
        return self.run_number

#get each agent class isolated and then access the well-being
    def get_agents_well_being(self, class_type):
        well_being_agents = []
        agents = self.agents_by_type[class_type].values()
        #print("length agents", len(agents))
        #print("mesa agent count", self.get_agent_count())

        for agent in agents:
            #print("agent id", agent.unique_id)
            well_being_agents.append(agent.well_being)


        #print("length well being", len(well_being_agents))
        try:
            well_being_average = sum(well_being_agents)/len(agents)
            #print("well being average", well_being_average)
            return(well_being_average)
        except:
            return(0)

    def get_agents_env_success(self, class_type, attribute):
        attribute_agents = []
        agents = self.agents_by_type[class_type].values()
        #print("length agents", len(agents))


        for agent in agents:
            #print("agent age schedule", agent.age)

            attribute_agents.append(agent.env_success)
            #print("env success", agent.env_success)


        #print("length success", len(success_agents))
        #print("list", sum(attribute_agents)/len(agents))
        try:
            attribute_average = sum(attribute_agents)/len(attribute_agents)
            #print("attribute_average", sum(attribute_agents))
            return(attribute_average)
        except:
            return(0)

    def get_agents_h_success(self, class_type, attribute):
        attribute_agents = []
        agents = self.agents_by_type[class_type].values()
        #print("length agents", len(agents))


        for agent in agents:
            #print("agent age schedule", agent.age)

            attribute_agents.append(agent.heritable_success)
            #print("env success", agent.env_success)


        #print("length success", len(success_agents))
        #print("list", sum(attribute_agents)/len(agents))
        try:
            attribute_average = sum(attribute_agents)/len(agents)
            #print("env success average", success_average)
            return(attribute_average)
        except:
            return(0)


    def get_agents_success(self, class_type):
        success_agents = []
        agents = self.agents_by_type[class_type].values()
        #print("length agents", len(agents))


        for agent in agents:
            #print("agent age schedule", agent.age)
            #print("h success schedule", agent.heritable_success)
            success_agents.append(agent.total_success)



        #print("length success", len(success_agents))
        try:
            success_average = sum(success_agents)/len(agents)
            #print("well being average", success_average)
            return(success_average)
        except:
            return(0)

    def get_agents_env_trauma(self, class_type, attribute):
        attribute_agents = []
        agents = self.agents_by_type[class_type].values()
        #print("length agents", len(agents))


        for agent in agents:
            #print("agent age schedule", agent.age)

            attribute_agents.append(agent.env_trauma)
            #print("env success", agent.env_success)


        #print("length success", len(success_agents))
        #print("list", sum(attribute_agents)/len(agents))
        try:
            attribute_average = sum(attribute_agents)/len(agents)
            #print("env success average", success_average)
            return(attribute_average)
        except:
            return(0)

    def get_agents_h_trauma(self, class_type, attribute):
        attribute_agents = []
        agents = self.agents_by_type[class_type].values()
        #print("length agents", len(agents))


        for agent in agents:
            #print("agent age schedule", agent.age)

            attribute_agents.append(agent.heritable_trauma)
            #print("env success", agent.env_success)


        #print("length success", len(success_agents))
        #print("list", sum(attribute_agents)/len(agents))
        try:
            attribute_average = sum(attribute_agents)/len(agents)
            #print("env success average", success_average)
            return(attribute_average)
        except:
            return(0)

    def get_agents_trauma(self, class_type):
        trauma_agents = []
        agents = self.agents_by_type[class_type].values()
        #print("length agents", len(agents))


        for agent in agents:
            #print("total trauma", agent.total_trauma)
            trauma_agents.append(agent.total_trauma)


        #print("length trauma", len(trauma_agents))
        try:
            trauma_average = sum(trauma_agents)/len(agents)
            #print("trauma agents", trauma_agents)
            #print("sum done", trauma_average)
            return(trauma_average)
        except:
            return(0)

    def get_leaders_tot_trauma(self, class_type):
        trauma_leaders = []
        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))

        #leader_count = 0
        for agent in agents:
            if agent.leader_status is 'yes':
                #leader_count = leader_count + 1
                trauma_leaders.append(agent.total_trauma)

        try:
            trauma_average = sum(trauma_leaders)/len(trauma_leaders)
            #print("trauma agents", trauma_agents)
            #print("sum done", trauma_average)
            return(trauma_average)
        except:
            return(0)

    def get_initial_num(self, model):
        return self.model.initial_num

    def get_leader_trauma_sensitivty_threshold(self, model):
        return self.model.leader_trauma_sensitivty_threshold

    def get_mean_heritable_succes(self, model):
         return self.model.mean_heritable_success
    def get_mean_heritable_trauma(self,model):
         return self.model.mean_heritable_trauma
    def get_mean_env_trauma(self,model):
         return self.model.mean_env_trauma
    def get_mean_env_success(self,model):
         return self.model.mean_env_success
    def get_mean_heritable_trau_sens(self,model):
         return self.model.mean_heritable_trau_sens
    def get_mean_hertiable_suc_sens(self,model):
         return self.model.mean_hertiable_suc_sens
    def get_mean_env_succ_sens(self,model):
         return self.model.mean_env_succ_sens
    def get_mean_env_trau_sens(self,model):
         return self.model.mean_env_trau_sens
    def get_heritable_ptsd_trauma_rate(self,model):
         return self.model.heritable_ptsd_trauma_rate
    def get_heritable_ptsd_suc_rate(self,model):
         return self.model.heritable_ptsd_suc_rate
    def get_leader_tot_trau_thres(self,model):
         return self.model.leader_tot_trau_thres
    '''def get_death_rate(self, model):
        return self.model.death_rate'''
    def get_trauma_heritable_proportion(self, model):
        return self.model.trauma_heritable_proportion
    def get_trauma_env_proportion(self, model):
        return self.model.trauma_env_proportion
    def get_success_heritable_proportion(self, model):
        return self.model.success_heritable_proportion
    def get_success_env_proportion(self, model):
        return self.model.success_env_proportion
    def get_leader_perc_val(self, model):
        return self.model.leader_perc
         #self.model.starting_variables


        #print("starting_variables", self.starting_variables)

    def get_leaders_tot_success(self, class_type):
        success_leaders = []
        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))

        #leader_count = 0
        for agent in agents:
            if agent.leader_status is 'yes':
                #leader_count = leader_count + 1
                success_leaders.append(agent.total_success)

        try:
            success_average = sum(success_leaders)/len(success_leaders)
            #print("success agents", success_agents)
            #print("sum done", success_average)
            return(success_average)
        except:
            return(0)


#now need a function for getting how many leaders, off spring and parents per step, and also how to get a step count thing'''
#a function that gets each agent values with unique_id, age, well_being'''
#add a table to the graph showing the starting variables'''
#print("full agents list", type(agents))
    def agent_age(self, class_type):
        #self.agents_age = []
        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))

        agetwocount = 0
        for agent in agents:
            if agent.age <= 2:
                agetwocount = agetwocount + 1
            #print(agent.age)
            #return agent.age
        return agetwocount

    def agent_avg_age(self, class_type):
        self.agents_age = []
        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))


        for agent in agents:

            #print(agent.age)
            self.agents_age.append(agent.age)

        return (sum(self.agents_age))/(len(self.agents_age))


    def agent_by_age(self, min_age, max_age, class_type):
        self.agents_by_age = []
        agents = self.agents_by_type[class_type].values()
        #print("length agent by age ======================", len(agents))


        if self.model.schedule.time < self.model.trauma_event:
            for agent in agents:
                #print(agent.age)
                if agent.age > min_age:
                    #print("passed min")
                    if agent.age < max_age:
                        #print("passed max")
                        #print("adding agent to list")
                        self.agents_by_age.append(agent.age)

            #print("agent by age list", self.agents_by_age)
            return sum(self.agents_by_age)

    def ptsd_percentage(self, class_type):
        count = 0
        agents = self.agents_by_type[class_type].values()
        #print("length agent by age ======================", len(agents))


        #if self.model.schedule.time < self.model.trauma_event:
        for agent in agents:
            #print(agent.age)
            if agent.total_trauma >= ptsd_value:
                #print("passed min")
                count = count + 1

            #print("agent by age list", self.agents_by_age)
            self.ptsd_percent = (count/len(agents))*100
            return (self.ptsd_percent)


    def agent_well_being(self, class_type):

        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))


        for agent in agents:
            #print("agent well being", agent.well_being)
            return agent.well_being


    def model_time(self):
        if self.time < trauma_event_fp_duration:
            print("model time is:", self.time)

    def leaders_count(self, class_type):

        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))

        leader_count = 0
        for agent in agents:
            if agent.leader_status is 'yes':
                leader_count = leader_count + 1
        return leader_count

    def leaders_perc(self, class_type):
        '''this transforms the % to 0-1 for graph purposes'''

        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))

        leader_count = 0
        for agent in agents:
            if agent.leader_status is 'yes':
                leader_count = leader_count + 1
        if leader_count > 0:
            leader_perc = (leader_count/len(agents))*100
        else:
            leader_perc = 0
        leader_perc = leader_perc/10
        return leader_perc

    def agent_env_success(self, class_type):
        #self._agents = self.agents_by_type[class_type]
        #print(self._agents)
        #print(type(self._agents))

        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))

        #print("length of agents list in schdule", len(agents))
        for agent in self.agents.values():
            return agent.age
            #print("succes printing:", agent.env_success)
        #  if agent.agent_type is 'agent':
        #        print("agent type agent")


    def agent_h_success(self, class_type):

        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))


        for agent in agents:
            #print(agent.success)
            return agent.heritable_success

    def agent_success(self, class_type):

        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))


        for agent in agents:
            #print(agent.success)
            return agent.total_success

    def agent_trauma(self, class_type):

        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))


        for agent in agents:
            #print(agent.trauma)
            return agent.total_trauma

    def leader_total(self, class_type):
        #self.agents_age = []
        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))


        for agent in agents:

            #print(agent.age)
            return agent.age

    def parent_total(self, class_type):

        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))


        for agent in agents:
            #print(agent.well_being)
            return agent.well_being

    def off_spring_total(self, class_type):

        agents = self.agents_by_type[class_type].values()
        #print("length", len(agents))


        for agent in agents:
            #print(agent.well_being)
            return agent.well_being
