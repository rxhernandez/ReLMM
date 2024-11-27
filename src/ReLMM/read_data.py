import pickle
import copy

import numpy as np
import pandas as pd

class Inputs:
    def __init__(self, input_type, input_path, input_file, add_target_noise = False):
        self.input_type = input_type
        self.input_path = input_path
        self.input_file = input_file
        self.add_target_noise = add_target_noise
        self.filename = self.input_path +'/'+ self.input_file
        
    def read_inputs(self):
        # if model_input.verbose:
        print('Reading data for the input dataset type: ', self.input_type)
        
        # Add options for different datasets that we want to read
        if self.input_type == 'PerovAlloys':
            x, y, descriptors = self.read_perovalloys()
        elif self.input_type == 'PALSearch':
            x, y, descriptors = self.read_palsearch()
        elif self.input_type == 'Gryffin':
            x, y, descriptors = self.read_gryffin()
        elif self.input_type == 'MPEA':
            x, y, descriptors = self.read_MPEA()
        elif self.input_type == 'MgAlloys':
            x, y, descriptors = self.read_MgAlloys()
        elif self.input_type == 'AlAlloys':
            x, y, descriptors = self.read_AlAlloys()
        elif self.input_type == 'COF':
            x, y, descriptors = self.read_COF()
        elif self.input_type == 'SynthData':
            x, y, descriptors = self.read_SynthData()
        else:
            raise ValueError('Input type not recognized')
        return x, y, descriptors
    
    def read_MPEA(self):
        '''
        This function reads the dataset from the HEA review paper: https://www.nature.com/articles/s41597-020-00768-9
        input_type='MPEA',
        input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/Space@Hopkins_HEA/dataset/',
        input_file='curated_MPEA_initial_training.csv'
        '''     
        data = pd.read_csv(self.filename)
 
        input_composition_cols = data.columns[0:29]
        descriptors = data.columns[30:35]
        input_composition_df = pd.DataFrame(data, columns=['Ti', 'Pd', 'Ga', 'Al', 'Co', 'Si', 'Mo', 'Sc', 'Zn', 'C', 'Sn', 'Nb', 'Ag', 'Mg', 'Mn', 'Y', 
                                    'Re', 'W', 'Zr', 'Ta', 'Fe', 'Cr', 'B', 'Cu', 'Hf', 'Li', 'V', 'Nd', 'Ni', 'Ca'])

        XX = pd.DataFrame(data, columns=data.columns[0:35])
        
        target = copy.deepcopy(data['Target'].to_numpy())
        YY = target.reshape(-1,1)

        return XX, YY, descriptors

    def read_perovalloys(self):
        '''
        This function reads the dataset from Maria Chan's paper: https://doi.org/10.1039/D1EE02971A
        It does not need any inputs. 
        Inputs to the class for this case are the default class inputs
        '''
        
        data = pd.read_csv(self.filename)

        Comp_desc = pd.DataFrame(data, columns=['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'Cl', 'Br', 'I'])

        # Descriptors using elemental properties (ionic radii, density etc.) shortlisted
        perov_desc_shortlist = pd.DataFrame(data, columns=['A_ion_rad', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_En', 
                                                'B_ion_rad', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_En', 
                                                'X_ion_rad', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_En', 'X_at_num'])

        # Descriptors using elemental properties (ionic radii, density etc.) shortlisted
        perov_desc_forNikhil = pd.DataFrame(data, columns=['A_ion_rad', 'A_EA', 'A_IE', 'A_En', 
                                                'B_ion_rad', 'B_EA', 'B_IE', 'B_En', 
                                                'X_ion_rad', 'X_EA', 'X_IE', 'X_En'])
        
        # Descriptors using elemental properties (ionic radii, density etc.) same as the paper
        perov_desc_paper = pd.DataFrame(data, columns=['A_ion_rad', 'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 
                                            'A_at_num', 'A_period', 'B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 
                                            'B_hov', 'B_En', 'B_at_num', 'B_period', 'X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 
                                            'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 'X_at_num', 'X_period'])
        
        # All descriptors
        All_desc = pd.DataFrame(data, columns=['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'Cl', 'Br', 'I', 'A_ion_rad', 
                                           'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 'A_at_num', 'A_period', 
                                           'B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 'B_hov', 'B_En', 'B_at_num', 
                                           'B_period', 'X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 
                                           'X_at_num', 'X_period'])
    
        descriptors = perov_desc_shortlist.columns
        XX = pd.DataFrame(data, columns=descriptors)
        # XX = copy.deepcopy(perov_desc_forNikhil.to_numpy())
        # descriptors = perov_desc_forNikhil.columns
    
        HSE_gap_copy = copy.deepcopy(data.Gap.to_numpy())
        YY = HSE_gap_copy.reshape(-1,1)
        
        return XX, YY, descriptors

    def read_palsearch(self):
        '''
        This function reads the datasets built for PAL-Search
        input_type='PALSearch',
        input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/PAL_Search/Datasets/',
        input_file='test_r2.xlsx'
        '''
        xls = pd.ExcelFile(self.filename)
        Data_DF1 = pd.read_excel(xls, 'ALL_RESULTS_DATA')
        Data_DF2 = pd.read_excel(xls, 'PROPERTY_BASKET')
        initial_cols = len(Data_DF1.columns)-1
        descriptors = []
        for num_col_df1 in range(0,initial_cols):
            bridge_dict = {}
            bridge = [element for element in Data_DF2.columns if Data_DF1.columns[num_col_df1] in element and 'CHOICES' not in element]
            descriptors.append(bridge)
            choices = [element for element in Data_DF2[Data_DF1.columns[num_col_df1]+'-CHOICES'] if str(element) != "nan"]
            list_features = []

            if bool(bridge):
                for column in bridge:
                    list_features_temp = Data_DF2[column][0:len(choices)].to_numpy().tolist()
                    list_features.append(list_features_temp)

                list_np = np.transpose(np.array(list_features))
                i=0
                for choice in choices:
                    bridge_dict[choice] = list_np[i]
                    i = i+1

                bridge_property_add = []
                for bridge_choice in Data_DF1[Data_DF1.columns[num_col_df1]]:

                    bridge_property_add_temp = bridge_dict[bridge_choice]
                    bridge_property_add.append(bridge_property_add_temp)   

                bridge_property_add = np.transpose(np.reshape(bridge_property_add,[len(Data_DF1[Data_DF1.columns[0]]),len(bridge)]))

                for column_num in range(0,len(bridge)):
                    Data_DF1[bridge[column_num]] = bridge_property_add[column_num]
        
        descriptors = np.array(list(itertools.chain.from_iterable(descriptors)),dtype='object')
        XX = pd.DataFrame(Data_DF1, columns=descriptors)

        target = copy.deepcopy(Data_DF1.Target.to_numpy())
        YY = target.reshape(-1,1)
        
        ## Adding noise to reactions 1-3 data
        if self.add_target_noise == 'True':
            if not "ALL_RESULTS_DATA_withNoise" in xls.sheet_names:
                YY_noise = np.random.normal(0,0.05,np.size(YY)).reshape(np.size(YY),1)
                YY = YY + YY_noise
                Data_DF1 = Data_DF1.drop(columns=['Target'])
                Data_DF1['Target'] = YY                
                writer = pd.ExcelWriter(self.filename, 'openpyxl', mode='a')
                Data_DF1.to_excel(writer, sheet_name='ALL_RESULTS_DATA_withNoise',index=False)
                writer.close()  
            elif "ALL_RESULTS_DATA_withNoise" in xls.sheet_names:
                Data_DF_target = pd.read_excel(xls, 'ALL_RESULTS_DATA_withNoise')
                target = copy.deepcopy(Data_DF_target.Target.to_numpy())
                YY = target.reshape(-1,1)
        
        return XX, YY, descriptors
    
    def read_gryffin(self):
        '''
        This function reads the perovskite dataset used in the GRYFFIN paper: https://doi.org/10.1063/5.0048164
        It does not need any inputs. 
        Inputs to the class for this case should be: 
        input_type='Gryffin',
        input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/Chemistry/SPIRAL/codes/RL/ReLMM/MHP_dataset/',
        input_file='perovskites.pkl'
        '''
        
        lookup_df = pickle.load(open(self.filename, 'rb'))
        perov_desc = pd.DataFrame(lookup_df, columns=['organic-homo',
                                   'organic-lumo', 'organic-dipole', 'organic-atomization',
                                   'organic-r_gyr', 'organic-total_mass', 'anion-electron_affinity',
                                   'anion-ionization_energy', 'anion-mass', 'anion-electronegativity',
                                   'cation-electron_affinity', 'cation-ionization_energy', 'cation-mass',
                                   'cation-electronegativity'])
        
        XX = perov_desc
        HSE_gap_copy = copy.deepcopy(lookup_df.hse06.to_numpy())
        YY=HSE_gap_copy.reshape(-1,1)
        YY = -1.0*YY
        
        return XX, YY, perov_desc.columns
    
    def read_MgAlloys(self):
        '''
        This function reads the dataset from the HEA review paper: https://www.nature.com/articles/s41597-020-00768-9
        input_type='MgAlloys',
        input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/Weiss_group_Sreenivas_dataset/V1_Aug1',
        input_file='Corrosion.csv'
        '''     
        data = pd.read_csv(self.filename)
 
        descriptors = data.columns[0:11]
        print(descriptors)
        XX = pd.DataFrame(data, columns=descriptors)        
        target = copy.deepcopy(data['Target'].to_numpy())
        YY = target.reshape(-1,1)

        return XX, YY, descriptors
    
    def read_AlAlloys(self):
        '''
        This function reads the dataset from the HEA review paper: https://www.nature.com/articles/s41597-020-00768-9
        input_type='AlAlloys',
        input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/Weihs_group_datasets/Jesse_colab/',
        input_file='AluminumFragmentDataforML.csv'
        '''     
        data = pd.read_csv(self.filename)
 
        descriptors = data.columns[0:10]
        XX = pd.DataFrame(data, columns=['Volume_Weighted_PSD_Mean','DOS','UTS','Elong_to_Failure','Modulus_of_Toughness','Normalized_MOT','YS'])        
        target = copy.deepcopy(data['D32'].to_numpy())
        YY = target.reshape(-1,1)

        return XX, YY, XX.columns 
    
    def read_COF(self):
        '''
        This function reads the dataset from the COF paper: https://pubs.acs.org/doi/10.1021/acs.chemmater.8b01425
        input_type='COF',
        input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/EngChem/Space@Hopkins_HEA/dataset/',
        input_file='COF.csv'
        '''     
        data = pd.read_csv(self.filename)
        descriptors = ['dimensions', 'bond_type', 'void_fraction', 'supercell_volume', 'density', 
                       'heat_desorption_high_P', 'absolute_methane_uptake_high_P', 
                       'absolute_methane_uptake_high_Pkg', 'excess_methane_uptake_high_P', 
                       'excess_methane_uptake_high_Pkg', 'heat_desorption_low_P', 
                       'absolute_methane_uptake_low_P', 'absolute_methane_uptake_low_Pkg', 
                       'excess_methane_uptake_low_P', 'excess_methane_uptake_low_Pkg', 'surface_area', 
                       'linkerA', 'linkerB', 'net', 'cell_a', 'cell_b', 'cell_c', 'alpha', 'beta', 'gamma', 
                       'num_carbon', 'num_fluorine', 'num_hydrogen', 'num_nitrogen', 'num_oxygen', 'num_sulfur', 
                       'num_silicon', 'vertices', 'edges', 'genus', 'largest_includedspherediameter', 
                       'largest_freespherediameter', 'freespherepathdiameter','absolute_methane_uptake_high_P',
                       'absolute_methane_uptake_low_P']
        
        XX = pd.DataFrame(data, columns=descriptors)
        target = copy.deepcopy(data['deliverable_capacity'].to_numpy())
        YY = target.reshape(-1,1)

        return XX, YY, descriptors
    
#     def read_SynthData(self):
#         '''
#         This function reads the dataset from the COF paper: https://pubs.acs.org/doi/10.1021/acs.chemmater.8b01425
#         input_type='SynthData',
#         input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/Chemistry/SPIRAL/codes/Seq-AE/datasets/synthetic_dataset/',
#         input_file='synthetic_data_nested_noisy_200.csv'
#         '''     
#         data = pd.read_csv(self.filename)
#         descriptors = ['z1','z2','z4','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13']
        
#         XX = pd.DataFrame(data, columns=descriptors)
#         target = copy.deepcopy(data['f14'].to_numpy())
#         YY = target.reshape(-1,1)

#         return XX, YY, descriptors

    def read_SynthData(self):
        '''
        This function reads the dataset from the COF paper: https://pubs.acs.org/doi/10.1021/acs.chemmater.8b01425
        input_type='SynthData',
        input_path='/Users/maitreyeesharma/WORKSPACE/PostDoc/Chemistry/SPIRAL/codes/Seq AE/datasets/synthetic_dataset/',
        input_file='synthetic_data_nested_noisy_200.csv'
        '''     
        data = pd.read_csv(self.filename)
        descriptors = ['z1','z2','z3','z4','z5','z6','z7','z8','f1','f2','f3','f4']
        
        XX = pd.DataFrame(data, columns=descriptors)
        target = copy.deepcopy(data['f5'].to_numpy())
        YY = target.reshape(-1,1)

        return XX, YY, descriptors