'''page to store values for run'''
#from schedule import RandomActivationByType
#from monte_carlo_generator import *


#what dataset are you using?
sample_or_no = str(input("Is this a sample? yes or no? ") or "no")#'no'
single_or_multi_str = str(input("Is this a single or multi-run? single or multi? ") or "multi") #'multi' or 'single'

#where to put this run output
#the below can be a list of parameters for batch run also
analysis_by_exp = "vertical_age_dpt" #["vertical_only", "vertical_age_dpt", "vertical_age_dpt_horizontal", "vertical_age_dpt_leaders"]

print("Is this running the sample? \n" + sample_or_no)
print("This is running for: \n" + analysis_by_exp)
while True:
    # some code here
    if input('Do You Want To Continue? y or n \n') != 'n':
        break

#SET tables
'''the multivariate analysis will indicate when the values start to stabilisie, looking at
variables against model_tick'''
warm_up_val = 75 #due to the PTSD rate being too high before that point, and the point at which values stabilise

'''single run'''
#the below are the same values but for single run
percent_change_val = 6 #degrading environmental trauma
standard_ag_rad_val = 2 #radius for agents search
leader_search_ag_rad_val = 6 #radius to look for leader
trauma_event_fp_duration_val = 50 #duration in which infants are extra effected by traumatic event
set_exp_age_val = 2 #exposure age
max_age_val = 34 #maximum age of agents
leader_e_trau_sens_thres_val = 4 #min threshold for trauma sensitivity
preg_min_age_val = 16 #min age for pregnancy
preg_perc_int_val = 7 #pregnancy chance
ptsd_value = 7 #tables - where on the scale is it considered ptsd
#GRID
grid_width = 200 #[100, 125, 150] #width
grid_height = 200 #[100, 125, 150] #height
compact_grid_width = 50
compact_grid_height = 50

#this is working for batch and single
model_iterations = 5

#single parameter run for testing
runs_model_var = 1 #300
initial_num_val = 1 #100

#br
model_max_steps = 5 #300

#scales are out out of 10
#increased env trauma and decreased heri to reflect that genetic adaptation were more prevalent
#in infants (which still exists)
mean_heritable_success_val= 5 #default 5 - this is to reflect a neautral value for success
mean_heritable_trauma_val= 1 # default 1 = to reflect the agents not having any trauma to start

mean_env_trauma_val= 10 #default 10 = the environment is traumatic
mean_env_success_val= 1 #default 1 = there is no success in the environment

#this is the divisable value for the percent change in value at every reproduction from parent to offspring
#assuming that the sensitivty to environmental gets less for trauma (more for succ), reflecting
#with each gen away from the primary event the potency is less
percent_change = [6] #[2, 6, 10] #produce_monte_carlo_range(mean_data=6, std=3, num_vals=4, int_or_float='type_int') #this is worked #[6]
#perc_change
# due to lack of research in this area these values are constant and slightly more arbitary
#these vals are engineered to go down during the course of the model assuming that things get better
#the further away from the traumatic event
mean_hertiable_suc_sens_val= 6
mean_heritable_trau_sens_val= 6

#this is linked to heritable_x positively
mean_env_succ_sens_val= 1
mean_env_trau_sens_val= 1

'''these value reflect the research into the proportion of genetic and heritable'''
success_heritable_proportion_val=65
success_env_proportion_val=35
trauma_heritable_proportion_val=71
trauma_env_proportion_val=29


#search radius for movement
off_spring_radius = 1
initial_agent_radius = 2
#leader_move_radius = 5

standard_ag_rad = [1]
leader_search_ag_rad = [6] #[5, 8] #produce_monte_carlo_range(mean_data=5, std=1, num_vals=4, int_or_float='type_int')
type_class = 'Agent_Class'

#mean_data=100 num_vals=3 std=25
init_initial_num= [100] #produce_monte_carlo_range(mean_data=100, std=25, num_vals=3, int_or_float='type_int')
mean_age = 13

'''the standard deviation defines the spread, so increase this number and it flattens the bell'''
#standard deviation
heri_std_scale_norm_dist = 1 #if this goes less than one then it becomes flatter

#studies show there is likely to be a different combination for heritability of trauma to cognitive ability
#num_vals=4
#the two below numbers have to total 100 so monte carlo is weird
success_heritable_proportion = 65
success_env_proportion = 35

#studies show there is likely to be a different combination for heritability of trauma to cognitive ability
#this value is based on social deprevation study, and its variablity ascribed to genetics
trauma_heritable_proportion = 71
trauma_env_proportion = 29

#the period of heightened susceptibility to long term ptsd based on studies
perc_after_exposure = 50 #was 50 changed to 30 to calibrate with validation literature
trauma_event_fp_duration = [25] #[25, 50] #50
set_exp_age = [2] #[2]
heritable_ptsd_trauma_rate = 10
heritable_ptsd_suc_rate = 1

#death
#lets get mortality rate specific as it is shown in the model that density of agents affects things
#owing to the fact that mortality rate is not constant over the 200+ years it will be kept at a constant
# but discuss that the value is based on historical mortality rates
#death_perc_int = 4 #this is based on contemporary studies into the chance of death in Benin (7.9/1000 or 0.79/100 p/yr) - used as a proxy for conditions of slavery
#this value takes in mind the transformations to the scale
#the below values create a survival of the fittest situation where those who fall below the proposed scale die :(
max_trauma = 10
min_success = 1
# and acknowledging that death rates are elusive for the time period
#death_rate_val = [0.79] #produce_monte_carlo_range(mean_data=0.79, std=0.1, num_vals=4, int_or_float='type_float')
max_age = [34] #17.8-38.1
#too_low_wb = 8

leader_e_trau_sens_thres = [2] #produce_monte_carlo_range(mean_data=4, std=1, num_vals=4, int_or_float='type_int') #lower threshold (the aim is to produce around 1-2% based on genius)
#this is setting the amount of leaders that can be created

leader_mean_heritable_success = 8
leader_mean_env_success = 8
leader_mean_heritable_trauma = 1
leader_mean_env_trauma = 1

#chance of pregnancy
#lets get this correct also to validate density
preg_min_age = [16] #produce_monte_carlo_range(mean_data=16, std=1, num_vals=4, int_or_float='type_int')
preg_perc_int = [7] #[5, 10] #produce_monte_carlo_range(mean_data=7, std=3, num_vals=4, int_or_float='type_int') #, 7
