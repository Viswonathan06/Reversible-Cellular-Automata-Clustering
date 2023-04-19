import numpy as np
import pandas as pd

class BiNCE:
    def __init__(self,dataset,length_of_each) -> None:   
        self.dataset_name =  'encoded_db'
        self.dataset_path = "../../Dataset"
        self.df = dataset
        self.save_path = '.temp/Dataset/fixed_width_encoding_'+self.dataset_name+'.csv'
        self.length_of_each = length_of_each
        self.even_regions = pow(self.length_of_each, 2)

    # normalize the dataset to [0, 1]
    def minmaxscale(self,series):
        min_val = series.min()
        max_val = series.max()
        return (np.divide((np.subtract(series, min_val,  dtype=np.float32)), (np.subtract(max_val,min_val, dtype=np.float32))))

    def evenly_spaced_array(self,num_splits):
        return np.linspace(0, 1, num_splits+1)[1:]
    
 
    def get_binary(self, arr2, number):
        idx = np.where(arr2 == number)
        return format(idx[0][0], '0{}b'.format(self.length_of_each))

    
    # Function to find the next greater element in an array for a given element
    def next_greater_element(self, x, arr):
        mask = arr >= x
        if any(mask):
            return arr[mask][0]
        else:
            return np.nan
        
    def encode(self):
        for cols in self.df.columns:
            self.df[cols] = self.minmaxscale(self.df[cols])
            self.df[cols] = self.df[cols].fillna(0)
        self.df = self.df.to_numpy()
        arr2 = self.evenly_spaced_array(self.even_regions)
        # Applying the function to each element in arr1
        new_df = []
        next_greater = []
        for arr1 in self.df:
            next_greater_arr = np.array([self.next_greater_element(x, arr2) for x in arr1])
            next_greater.append(next_greater_arr)
            binarized_next = []
            for number in next_greater_arr:
                binarized_next.append(self.get_binary(arr2,number))
            new_df.append(binarized_next)
        i=0
        for strs in new_df:
            new_df[i] = ''.join(new_df[i])
            i+=1
        new_df = pd.DataFrame(new_df, dtype=str)
        return new_df
  

    