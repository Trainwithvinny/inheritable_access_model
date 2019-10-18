import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from model import MoneyModel
import seaborn  as sns
#import adjustable_values
import os
from adjustable_values import *
#from run import *
import statistics
from sklearn import preprocessing
import scipy
import statsmodels.api as sm
from scipy.stats import ttest_1samp

class Data_Visualisation(object):


    def __init__(self):
        #sample_or_no = 'no'
        single_or_multi = single_or_multi_str

        self.my_path = os.path.abspath(__file__)
        self.mother_folder, data_vis_py = os.path.split(self.my_path)
        self.grouped_folder, models = os.path.split(self.mother_folder)
        self.model_output_folder = os.path.join(self.grouped_folder, "model_output")
        self.data_folder = os.path.join(self.model_output_folder, analysis_by_exp)
        print("data folder is: ", self.data_folder)
        self.linear_regression_folder = os.path.join(self.data_folder, "linear_regression")
        self.lineplot_folder = os.path.join(self.data_folder, "lineplot_folder")
        self.boxplot_folder = os.path.join(self.data_folder, "boxplot_folder")
        self.coefficients_folder = os.path.join(self.data_folder, "coefficients_folder")
        self.multivariate_folder = os.path.join(self.data_folder, "multivariate_folder")
        self.csv_folder = os.path.join(self.data_folder, "csv_folder")
        self.csv_folder_500 = os.path.join(self.data_folder, "csv_folder_(500yr_04_10_19)")
        self.sa_folder = os.path.join(self.data_folder, "sa_folder")
        self.single_run_output = os.path.join(self.data_folder, "single_run_output")


        print("my path", self.my_path)
        print("my data folder", self.data_folder)

        self.filename2 = "sample_" + analysis_by_exp + ".csv"
        self.filename17 = "self.whole_df_" + analysis_by_exp + ".csv"
        self.filename18 = "self.group_by_run_" + analysis_by_exp + ".csv"
        self.filename19 = "self.warm_up_table_" + analysis_by_exp + ".csv"
        self.filename20 = "self.single_run_" + analysis_by_exp + ".csv"
        self.filename21 = "step_data_batch_hoz_" + analysis_by_exp + ".csv"
        self.filename22 = analysis_by_exp + ".csv"
        self.filename23 = "self.multivariate_table_" + analysis_by_exp + ".csv"
        self.filename24 = "self.tick_run_" + analysis_by_exp + ".csv"
        self.filename25 = "data_agent_" + analysis_by_exp + ".csv"

        self.key_data_points = (r'initial_n=' + str(init_initial_num),
        r'trauma_heritable_proportion=' + str(trauma_heritable_proportion),
        r'trauma_env_proportion =' + str(trauma_env_proportion),
        r'success_heritable_proportion=' + str(success_heritable_proportion),
        r'success_env_proportion =' + str(success_env_proportion),
        r'mean_heritable_success=' + str(mean_heritable_success_val) ,
        r'mean_heritable_trauma=' + str(mean_heritable_trauma_val) ,
        r'mean_env_trauma=' + str(mean_env_trauma_val) ,
        r'mean_env_success=' + str(mean_env_success_val) ,
        r'mean_env_succ_sens =' + str(mean_env_succ_sens_val) ,
        r'mean_env_trau_sens =' + str(mean_env_trau_sens_val) ,
        r'leader_e_tot_trau_thres =' + str(leader_e_trau_sens_thres_val),
        r'percent_change_val =' + str(percent_change_val),
        r'leader_search_ag_rad =' + str(leader_search_ag_rad_val),
        r'ptsd_value =' + str(ptsd_value),
        r'perc_after_exposure =' + str(perc_after_exposure),
        r'trauma_event_fp_duration =' + str(percent_change_val),
        r'max_age=' + str(max_age),
        r'mean_age=' + str(mean_age),
        r'preg_age=' + str(preg_min_age),
        )

        whole_df = pd.read_csv(os.path.join(self.csv_folder, self.filename21), index_col=0)
        #self.whole_df = pd.read_csv(os.path.join(self.csv_folder, self.filename21), index_col=0)
        #self.whole_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        self.whole_df = pd.DataFrame(data=whole_df)
        self.whole_df = self.whole_df.round(2)

        my_list = []
        length = len(self.whole_df)

        n = 0
        #print("length", range(length))
        for x in range(length):
            n = n + 1
            for i in range(model_max_steps):
                #n = n+1
                my_list.append(n)
                continue
        my_list = my_list[:length]

        self.whole_df['run'] = my_list

        #create a sample df
        if sample_or_no is 'sample':
            sample = self.whole_df.sample(frac=0.01, replace=True, random_state=1)
            self.sample = pd.DataFrame(data=sample)
            self.whole_df = self.sample
            self.sample.to_csv(os.path.join(self.csv_folder, self.filename2))
        if sample_or_no is 'no':
            self.whole_df.to_csv(os.path.join(self.csv_folder, self.filename17))

        if single_or_multi is 'single':
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
            self.whole_df['scaled_well_being']  = min_max_scaler.fit_transform(self.whole_df[['agent_well_being']])
            #find agents who lived range(33)
            self.group_by_id = self.whole_df.groupby(by=['AgentID'], group_keys=True)

            #self.age_iq_table = self.whole_df.loc[self.whole_df['agent_age'].isin([11])] & self.whole_df.loc[self.whole_df['agent_age'].isin([33])]
            #self.age_iq_table = self.whole_df[(self.whole_df['agent_age']==11) & (self.whole_df['agent_age']==33)]
            iq_11 = self.whole_df.groupby('AgentID')['total_success'].nth(10)
            iq_33 = self.whole_df.groupby('AgentID')['total_success'].nth(32)
            self.age_iq_table = pd.concat([iq_11, iq_33], axis = 1, ignore_index=False).dropna()
            self.age_iq_table = self.age_iq_table.reset_index(drop=False)
            self.age_iq_table.columns = ["AgentID", "success_11", "success_33"]
            #self.age_iq_table



        if single_or_multi_str is 'multi':
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
            self.whole_df['scaled_well_being']  = min_max_scaler.fit_transform(self.whole_df[['average_well_being']])
            scale_whole  = min_max_scaler.fit_transform(self.whole_df.values)


            self.scaled_df = pd.DataFrame(scale_whole, columns = self.whole_df.columns)
            self.scaled_df.to_csv(os.path.join(self.csv_folder, "self.scaled_df.csv"))
            self.mean_table = self.whole_df.groupby('model_tick').mean()
            self.median_table = self.whole_df.median()
            self.sd_table = self.whole_df.std()
            self.sd_table = self.sd_table.round()
            self.group_by_run = self.whole_df.groupby(by=['run']).mean()
            self.group_by_run.to_csv(os.path.join(self.csv_folder, self.filename18))

            self.tick_run = self.whole_df.groupby(by=['model_tick']).mean()
            self.tick_run.to_csv(os.path.join(self.csv_folder, self.filename24))


            self.warm_up_table = self.whole_df[self.whole_df.model_tick > warm_up_val]
            self.warm_up_table.to_csv(os.path.join(self.csv_folder, self.filename19))

            self.key_tick_point_tbl = self.whole_df.loc[self.whole_df['model_tick'].isin([50,75,100,125,150,175,200,225,250])]

            #self.single_run = self.warm_up_table.loc[self.warm_up_table['run'].isin([1])]
            #self.single_run.to_csv(os.path.join(self.csv_folder, self.filename20))


            self.multivariate_table = self.warm_up_table.drop(['mean_env_success', 'mean_env_trauma',
                        'mean_env_succ_sens', 'mean_env_trau_sens','agent_by_age_2',
                        'mean_heritable_trauma','mean_heritable_success',
                        'mean_hertiable_suc_sens', 'mean_heritable_trau_sens', 'initial_num', 'run',
                        'success_heritable_proportion', 'success_env_proportion',
                        'trauma_heritable_proportion', 'trauma_env_proportion', 'percent_change',
                        'agent_radius', 'leader_search_ag_rad', 'event_duration',
                        'set_exp_age', 'max_age', 'leader_trau_thres',
                        'min_preg_age', 'preg_chance', 'death_percent', 'model_tick'], axis = 1)
            #self.multivariate_table.replace(r'\s+', np.nan, regex=True, inplace=True)
            #self.multivariate_table['model_tick'] = self.multivariate_table.index

            self.multivariate_table.to_csv(os.path.join(self.csv_folder, self.filename23))




    def exploratory_output(self, data_set, title_string, vars_list=None):


        '''multivariate analysis of all data points to see relationship'''
        fig = plt.figure(figsize=(20,20))
        sns.pairplot(data_set, diag_kind="kde", kind="reg",
        vars=vars_list)
        plt.savefig(os.path.join(self.multivariate_folder, title_string))
        #plt.show()


    def covariance(self, xy_dict):
        '''correlation'''
        '''covariance - if its positive or negative, but you cant tell much more than that'''

        with open(os.path.join(self.coefficients_folder, "coefficients.txt"), 'w+') as r_o_t:
            covariance_info = "covariance(a,a)  covariance(a,b)"
            covariance_info2 = "covariance(a,b)  covariance(b,b)"

            r_o_t.writelines('\n' + (covariance_info))
            r_o_t.writelines('\n' + (covariance_info2) + '\n')

        #list_covariance = []

            for k,v in xy_dict.items():

                X = k
                Y = v

                covariance = np.cov(self.multivariate_table[X], self.multivariate_table[Y])
                covariance_title = "correlation of " + str(X).upper() + " and " + str(Y).upper()

                r_o_t.writelines('\n' + str(covariance_title) + '\n')
                r_o_t.writelines('\n' + str(covariance) + '\n')
                r_o_t.writelines('\n')
                r_o_t.writelines('\n')
        r_o_t.close()



    def pearson_corr(self, xy_dict, folder, data_set):


        with open(os.path.join(folder, "pearson_coefficients.txt"), 'w+') as r_o_t:
            info = """pearson r \n correlation coefficient & p-value \n compare the p-value to the significance levels
            (0.05 eg the risk of incorrect is 5%) 0 means no relationship, lesser means relationship (e- == negative number)\n"""

            r_o_t.writelines('\n' + (info) + '\n\n\n\n')


            for k,v in xy_dict.items():

                X = k
                Y = v
                #data1 = data_set[X].where(data_set["agent_age"]==11, other=0)
                #data2 = data_set[X].where(data_set["agent_age"]==33, other=0)
                #print(data1, "\n", data2)
                stat = scipy.stats.pearsonr(data_set[X], data_set[Y])
                print("stat", stat)
                title = "correlation of " + str(X).upper() + " and " + str(Y).upper()

                r_o_t.writelines('\n' + str(title) + '\n')
                r_o_t.writelines('\n' + str(stat) + '\n')
                r_o_t.writelines('\n')
                r_o_t.writelines('\n')
        r_o_t.close()

    def spearmans_corr(self, xy_dict, data_set):
        the_string = str(xy_dict)
        splitted = the_string.split()
        save_title = str(splitted[:8])
        with open(os.path.join(self.coefficients_folder, save_title + "_coefficients.txt"), 'w+') as r_o_t:
            info = '''the relationship between the two variables may vary and a non-Gaussian distribution
            - the strength between the two samples - linear relationship not assumed, a monotonic one is
            (math name for inc or dec relationship between the two variables)'''

            r_o_t.writelines('\n' + (info) + '\n\n\n\n')


            for k,v in xy_dict.items():

                X = k
                Y = v

                stat = scipy.stats.spearmanr(scipy.stats.pearsonr(data_set[X], data_set[Y]))
                title = "correlation of " + str(X).upper() + " and " + str(Y).upper()

                r_o_t.writelines('\n' + str(title) + '\n')
                r_o_t.writelines('\n' + str(stat) + '\n')
                r_o_t.writelines('\n')
                r_o_t.writelines('\n')
        r_o_t.close()

    def analytical_output_shaded_plots(self, X, Y1, data_set, folder, Y2=None, Y3=None, name_suffix=None):


        fig = plt.figure(figsize=(15,10))

        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

        sns.lineplot(x=X, y=Y1, data=data_set, label=Y1, color="blue")

        if Y2 is not None:
            sns.lineplot(x=X, y=Y2, data=data_set, label=Y2, color="red")
            #ax.text(200, 7, r'success s.d=' + str(self.sd_table[Y2]))
            ax.axhline(y=self.median_table[Y2], linewidth=1.5, linestyle=':', color='red')

        if Y3 is not None:
            sns.lineplot(x=X, y=Y3, data=data_set, label=Y3, color="green")
            #ax.text(200, 6, r'success s.d=' + str(self.sd_table[Y3]))
            ax.axhline(y=self.median_table[Y3], linewidth=1.5, linestyle=':', color='green')

        plt.legend(loc='best') #labels=[1, Y2, Y3],
        #ax.text(200, 5, s=r'trauma s.d=' + str(self.sd_table[Y1]))
        ax.axhline(y=self.median_table[Y1], linewidth=1.5, linestyle=':', color='blue')

        textstr = '\n'.join((
            self.key_data_points
            ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(330, 2, textstr, fontsize=8, bbox=props)
        ax.grid(True)

        if Y3 and Y2 is None:
            title = str(X).upper() + '_' + str(Y1).upper() + "_" + str(Y2).upper()
        if Y1 and Y2 is not None:
            title = str(X).upper() + '_' + str(Y1).upper() + "_" + str(Y2).upper()
        else:
            title = str(X).upper() + '_' + str(Y1).upper() + "_" + str(Y2).upper() + "_" + str(Y3).upper()

        ax.set_title(title)
        #ax.set_ylabel('Mean values')
        ax.set_xlabel(X)
        plt.savefig(os.path.join(folder, (name_suffix + title)))
        #plt.show()




    def analytical_output_boxplots(self, xy_dict, data_set, folder):

        for k,v in xy_dict.items():

            X = k
            Y = v
            fig = plt.figure(figsize=(15,10))
            sns.set(style="whitegrid")

            ax = sns.boxplot(x=X, y=Y, data=data_set)
            title = str(X).upper() + '_' + str(Y).upper()
            plt.savefig(os.path.join(folder, title))
        print("boxplot done")
            #plt.show()


    def do_linear_regression_models(self,xy_dict, folder, data_set, name_suffix=None):

        print(type(xy_dict), len(xy_dict))

        for k,v in xy_dict.items():

            X = k
            Y = v
            fig = plt.figure(figsize=(15,10))

            ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

            sns.regplot(x=X, y=Y, data=data_set);

            textstr = '\n'.join((
                    self.key_data_points
                    ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(300, 8.5, textstr, fontsize=8, bbox=props)
            ax.grid(True)
            '''the below shows how to add arrows etc to specific points on the plot'''
            title = str(X).upper() + "_v_"+ str(Y).upper()
            ax.set_title(title)
            ax.set_ylabel(Y)
            ax.set_xlabel(X)
            plt.savefig(os.path.join(folder, (name_suffix + title)))
            #plt.show()

    def analytical_output_regression(self, xy_dict, data_set, name_suffix=None):

        with open(os.path.join(self.linear_regression_folder, "regression_info.txt"), 'w+') as r_o_t:
            r_o_t.writelines('''Df of residuals and models relates to the degrees of freedom —
            “the number of values in the final calculation of a statistic that are free to vary.\n
            \ncoefficient indicates that as the independent variable increases by 1,
            the dependent variable increases by the coefficient - note that the coefficient is
            different without the constant\n
            \n  R squared - the percentage of variance our model explains (the more variables in X, the higher R sq is);
            the standard error std of the sampling distribution of a statistic, most commonly of the mean\n''' )
        r_o_t.close()

        for k,v in xy_dict.items():

            X = k
            Y = v

            self.X = X
            self.Y = Y
            independent_var = data_set[X] #this can have more than one variable X = df[[“RM”, “LSTAT”]]
            dependent_var = data_set[Y]
            #first without a constant
            X = independent_var
            Y = dependent_var #what we are trying to predict
            X = sm.add_constant(X) #this adds an intercept (beta_0) to the model
            '''Y is output and X is output'''
            model1 = sm.OLS(Y, X).fit() #ols Ordinary Least Squares - least squares means we trying
            #fit a a regression line that would minimise the square of distance from the regression line
            title = str(self.X).upper() + "_v_" + str(self.Y).upper()
            predictions = model1.predict(X) # make the predictions by the model - should this be printed?
            #return model1.summary() #print
            with open(os.path.join(self.linear_regression_folder, name_suffix + title + "_" + analysis_by_exp + ".txt"), 'w+') as r_o_t:
                r_o_t.writelines(str(model1.summary())+ '\n')
                #r_o_t.writelines(str(predictions)+ '\n')

            r_o_t.close()


    def p_value_check_ind(self, expected_mean_and_samp_pop_dict, expected_data_set, samp_data_set):
        '''i think all the data for this project falls under related dependent
        ttest'''
        #alternative hypothesis and null
        #this is for independent variables meaning the two samples are unrelated
        #there should be a continuous dependent variable and one independent categorial variable
        #that has two levels/groups
        #the null hypothesis is that the averages should be identical
        with open(os.path.join(folder, "p_values.txt"), 'w+') as r_o_t:
            info = '''When talking statistics, a p-value for a statistical model is the probability
            that when the null hypothesis is true, the statistical summary is equal to or greater than
            the actual observed results. This is also termed probability value or asymptotic significance.'''


            r_o_t.writelines('\n' + (info) + '\n\n\n\n')


            for k,v in expected_mean_and_samp_pop_dict.items():

                X = k
                Y = v

                data1 = data_set['scaled_well_being'].where(data_set['preg_chance'] == 5, other = 0)
                data2 = data_set['scaled_well_being'].where(data_set['preg_chance'] == 10, other = 0)

                stats.ttest_ind(a= data1, b= data2)


                title = "Significance of expected mean of " + str(X).upper() + " compared to the sample population: " + str(Y).upper()

                r_o_t.writelines('\n' + str(title) + '\n')
                #r_o_t.writelines('\n' + p_v_outcome + '\n')
                #r_o_t.writelines('\n' + str(pval1) + '\n')
                r_o_t.writelines('\n')
                r_o_t.writelines('\n')
        r_o_t.close()



    def p_value_check(self, X, hue_cat, hue1, hue2, folder, data_set):
        #alternative hypothesis and null
        #this is for dependent variables, two dataset of the same population but with
        #different settings, they are related
        #the null hypothesis is that the means are equal - if its significant we can reject this
        #does the overall average in one setting differ greatly to that of another
        with open(os.path.join(folder, hue_cat + X + "p_values.txt"), 'w+') as r_o_t:
            info = '''When talking statistics, a p-value for a statistical model is the probability
            that when the null hypothesis is true, the statistical summary is equal to or greater than
            the actual observed results. This is also termed probability value or asymptotic significance.'''


            r_o_t.writelines('\n' + (info) + '\n\n\n\n')

            data1 = data_set[X].where(data_set[hue_cat] == hue1, other = 0)
            data2 = data_set[X].where(data_set[hue_cat] == hue2, other = 0)
            #data1 = 5 * np.random.randn(100) + 50
            #data2 = 5 * np.random.randn(100) + 51
            # calculate means
            pval_ttest = stats.ttest_rel(a=data1, b=data2, nan_policy='omit')
            #print(pval_ttest)


            title = "Significance of expected mean of " + str(hue_cat).upper() + " compared to the sample population: " + str(X).upper()

            r_o_t.writelines('\n' + str(title) + '\n')
            r_o_t.writelines('\n' + str(pval_ttest) + '\n')
            #r_o_t.writelines('\n' + str(pval1) + '\n')
            r_o_t.writelines('\n')
            r_o_t.writelines('\n')
        r_o_t.close()


    def run_exploratory_analysis(self):

        self.exploratory_output(data_set= self.multivariate_table, title_string= "multivariate_whole", vars_list=None)
        #self.exploratory_output(data_set= self.multivariate_table, title_string= "multivariate_well_being", vars_list= ["average_well_being", "agent_count", "model_tick", "agent_avg_age"])



    def run_analytics(self):

        #integrated already
        #500 yrs
        #self.analytical_output_shaded_plots(folder=self.lineplot_folder, X="model_tick", Y1="leader_perc", data_set=self.warm_up_table, Y2="ptsd_percent", Y3="scaled_well_being", name_suffix = '500yrs')
        #self.analytical_output_shaded_plots(folder=self.lineplot_folder, X="model_tick", Y1="pop_growth_rate", data_set=self.warm_up_table, Y2="death_percent", name_suffix="500yr", Y3="leader_perc")


        #integrated already
        #self.analytical_output_regression(xy_dict={"average_well_being": "pop_growth_rate"}, data_set=self.warm_up_table)
        #self.analytical_output_regression(xy_dict={"ptsd_percent": "average_well_being"}, data_set=self.warm_up_table)
        #self.analytical_output_regression(xy_dict={"ptsd_percent": "leader_perc"}, data_set=self.warm_up_table)
        #self.analytical_output_regression(xy_dict={"leader_perc": "average_well_being"}, data_set=self.warm_up_table)
        ###############

        #integrated already
        #trauma/iq
        #self.analytical_output_regression(xy_dict={"average_total_trauma": "average_total_success"}, data_set=self.warm_up_table)
        #self.do_linear_regression_models(xy_dict={"average_total_trauma": "average_total_success"}, data_set=self.warm_up_table, folder=self.linear_regression_folder)
        #self.analytical_output_regression(xy_dict={"success_11": "success_33"}, data_set=self.age_iq_table, name_suffix="single_run")
        #self.do_linear_regression_models(xy_dict={"success_11": "success_33"}, data_set=self.age_iq_table, folder=self.linear_regression_folder, name_suffix="single_run")
        #self.analytical_output_regression(xy_dict={"total_trauma": "total_success"}, data_set=self.whole_df, name_suffix="single_run")
        #self.do_linear_regression_models(xy_dict={"total_trauma": "total_success"}, data_set=self.whole_df, folder=self.linear_regression_folder, name_suffix="single_run")


        #heritability or environment
        #self.analytical_output_regression(xy_dict={"average_h_trauma": "average_well_being"}, data_set=self.warm_up_table, name_suffix="h_or_v")
        #self.do_linear_regression_models(xy_dict={"average_h_trauma": "average_well_being"}, data_set=self.warm_up_table, folder=self.linear_regression_folder, name_suffix="h_or_v")

        #self.analytical_output_regression(xy_dict={"average_h_success": "average_well_being"}, data_set=self.warm_up_table, name_suffix="h_or_v")
        #self.do_linear_regression_models(xy_dict={"average_h_success": "average_well_being"}, data_set=self.warm_up_table, folder=self.linear_regression_folder, name_suffix="h_or_v")

        #self.analytical_output_regression(xy_dict={"average_env_trauma": "average_well_being"}, data_set=self.warm_up_table, name_suffix="h_or_v")
        #self.do_linear_regression_models(xy_dict={"average_env_trauma": "average_well_being"}, data_set=self.warm_up_table, folder=self.linear_regression_folder, name_suffix="h_or_v")

        #self.analytical_output_regression(xy_dict={"average_env_success": "average_well_being"}, data_set=self.warm_up_table, name_suffix="h_or_v")
        #self.do_linear_regression_models(xy_dict={"average_env_success": "average_well_being"}, data_set=self.warm_up_table, folder=self.linear_regression_folder, name_suffix="h_or_v")


        #self.analytical_output_regression(xy_dict={"average_h_trauma": "ptsd_percent"}, data_set=self.warm_up_table, name_suffix="2")
        #self.do_linear_regression_models(xy_dict={"average_h_trauma": "ptsd_percent"}, data_set=self.warm_up_table, folder=self.linear_regression_folder, name_suffix="2")

        #self.analytical_output_regression(xy_dict={"average_h_success": "ptsd_percent"}, data_set=self.warm_up_table, name_suffix="2")
        #self.do_linear_regression_models(xy_dict={"average_h_success": "ptsd_percent"}, data_set=self.warm_up_table, folder=self.linear_regression_folder, name_suffix="2")

        #self.analytical_output_regression(xy_dict={"average_env_trauma": "ptsd_percent"}, data_set=self.warm_up_table, name_suffix="2")
        #self.do_linear_regression_models(xy_dict={"average_env_trauma": "ptsd_percent"}, data_set=self.warm_up_table, folder=self.linear_regression_folder, name_suffix="2")

        #self.analytical_output_regression(xy_dict={"average_env_success": "ptsd_percent"}, data_set=self.warm_up_table, name_suffix="2")
        #self.do_linear_regression_models(xy_dict={"average_env_success": "ptsd_percent"}, data_set=self.warm_up_table, folder=self.linear_regression_folder, name_suffix="2")

        #self.analytical_output_regression(xy_dict={"average_total_trauma": "ptsd_percent"}, data_set=self.warm_up_table, name_suffix="___")
        #self.do_linear_regression_models(xy_dict={"average_total_trauma": "ptsd_percent"}, data_set=self.warm_up_table, folder=self.linear_regression_folder, name_suffix="___")

        #integrated already
        #self.analytical_output_shaded_plots(folder=self.lineplot_folder, X="model_tick", Y1="average_total_trauma", data_set=self.warm_up_table, Y2="scaled_well_being", name_suffix="300yr", Y3="average_total_success")
        #self.analytical_output_shaded_plots(folder=self.lineplot_folder, X="model_tick", Y1="pop_growth_rate", data_set=self.warm_up_table, Y2="death_percent", name_suffix="300yr", Y3="leader_perc")
        #self.analytical_output_boxplots(xy_dict={"model_tick": "ptsd_percent"}, data_set=self.key_tick_point_tbl, folder=self.boxplot_folder)
        #self.analytical_output_boxplots(xy_dict={"model_tick": "scaled_well_being"}, data_set=self.key_tick_point_tbl, folder=self.boxplot_folder)
        #self.analytical_output_boxplots(xy_dict={"model_tick": "average_well_being"}, data_set=self.key_tick_point_tbl, folder=self.boxplot_folder)
        #############

        #self.analytical_output_shaded_plots(folder=self.lineplot_folder, X="model_tick", Y1="leader_perc", data_set=self.warm_up_table, Y2="scaled_well_being", Y3=None)
        #self.analytical_output_shaded_plots(folder=self.lineplot_folder, X="model_tick", Y1="average_total_trauma", data_set=self.warm_up_table, Y2="average_total_success", Y3="leader_perc")
        #self.analytical_output_shaded_plots(folder=self.lineplot_folder, X="model_tick", Y1="scaled_well_being", data_set=self.warm_up_table, Y2=None, Y3="leader_perc")
        #self.analytical_output_shaded_plots(folder=self.lineplot_folder, X="model_tick", Y1="average_total_trauma", Y2="average_total_success", data_set=self.warm_up_table, Y3="leader_perc")


        #self.analytical_output_boxplots(xy_dict={"model_tick": "average_total_trauma"}, data_set=self.key_tick_point_tbl, folder=self.boxplot_folder)
        #self.analytical_output_boxplots(xy_dict={"model_tick": "average_total_success"}, data_set=self.key_tick_point_tbl, folder=self.boxplot_folder)

        #vertical specific
        #self.analytical_output_shaded_plots(folder=self.lineplot_folder, X="model_tick", Y1="average_total_trauma", Y2="average_total_success", data_set=self.warm_up_table, Y3="ptsd_percent")
        #self.analytical_output_shaded_plots(folder=self.lineplot_folder, X="model_tick", Y1="average_total_success", Y2="average_env_success", data_set=self.warm_up_table, Y3="average_h_success")
        #self.analytical_output_shaded_plots(folder=self.lineplot_folder, X="model_tick", Y1="ptsd_percent", Y2="average_env_success", data_set=self.warm_up_table, Y3="average_h_success")
        #self.analytical_output_boxplots(xy_dict={"model_tick": "average_well_being"}, data_set=self.key_tick_point_tbl, folder=self.boxplot_folder)

        #integrated already
        #self.do_linear_regression_models(xy_dict={"average_well_being": "pop_growth_rate"}, data_set=self.warm_up_table, folder=self.linear_regression_folder)
        #self.do_linear_regression_models(xy_dict={"ptsd_percent": "average_well_being"}, data_set=self.warm_up_table, folder=self.linear_regression_folder)
        #self.do_linear_regression_models(xy_dict={"ptsd_percent": "leader_perc"}, data_set=self.warm_up_table, folder=self.linear_regression_folder)
        #self.do_linear_regression_models(xy_dict={"leader_perc": "average_well_being"}, data_set=self.warm_up_table, folder=self.linear_regression_folder)
        ##########
        #self.spearmans_corr(xy_dict=non_gaus_corr1, data_set=self.warm_up_table)
        #self.spearmans_corr(xy_dict=non_gaus_corr2, data_set=self.warm_up_table)
        #self.spearmans_corr(xy_dict=non_gaus_corr3, data_set=self.warm_up_table)
        #self.spearmans_corr(xy_dict=non_gaus_corr4, data_set=self.warm_up_table)
        #self.pearson_corr(xy_dict={"success_11":"success_33"}, folder=self.coefficients_folder, data_set=self.age_iq_table)
        #self.pearson_corr(xy_dict={"average_total_trauma":"average_total_success"}, folder=self.coefficients_folder, data_set=self.whole_df)
        #self.p_value_check(expected_mean_and_samp_pop_dict={"average_well_being": "agent_count", "average_well_being": "model_tick"}, data_set=self.warm_up_table)
