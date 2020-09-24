# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:33:41 2020

@author: Izabela Perenc

NN_Manager allows to save and follow work progress while training NNs.
The key concept is to create a model which can be trained with different 
configuration settings. Summary of all experiments can be easily acessed.

TODO:
    - saving a model 
    - generation of configuration files
    - running not executed generation files
    - preparing summary
"""

import copy
import datetime
from   keras.models import model_from_json
import os
import shutil

def is_iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True
    
def strip_model_config(model_config):
    model_config = copy.deepcopy(model_config)
    recursive_key_removal(model_config, 'name')
    recursive_key_removal(model_config, 'inbound_nodes')
    #recursive_key_removal(model_config, 'name')
    model_config.pop('input_layers', None)
    model_config.pop('output_layers', None)
    return model_config  
  
def recursive_key_removal(dictionary, key):
    if not is_iterable(dictionary):
        return
    if isinstance(dictionary, dict):
        dictionary.pop(key, None)
        for k in dictionary.keys():
            recursive_key_removal(dictionary[k], key)
    else:
        for elem in dictionary:
            recursive_key_removal(elem, key)
    
      
        
class NN_Manager:
    
    def __init__(self, path = r".\DATA\\"):
        self.directory = path
        self.existing_models = []
        subdirs = self._get_all_subdirs()
        print("NN_Manager works in directory containing {} models".format(len(subdirs)))
           
    #save model - creates folder, verifies if same model already exists
    def save_model(self, model, allow_duplicates=False):
        subdirs = self._get_all_subdirs()
        existing_models_configs = [self.load_model(sd).get_config() for sd in subdirs]
        #trzeba zignorować nazwy
        #my_dict.pop('key', None)
        if not allow_duplicates:
            for sd in subdirs:
                c1 = self.load_model(sd).get_config()
                c2 = model.get_config()
                if  strip_model_config(c1)==strip_model_config(c2) :
                    print("The same model already exists!")
                    return None
        model_json = model.to_json()
        filename = self._generate_model_filename(model)
        print("Saving: "+filename)
        model_directory = self.directory + "\\" + filename + "\\"
        try:
            os.mkdir(model_directory)
        except OSError:
            print ("Creation of the directory %s failed" % model_directory)
        with open(model_directory + filename + ".json", "w") as json_file:
            json_file.write(model_json)
        return filename
    
    def load_model(self, filename):
        with open(self.directory + "\\" + filename + "\\" + filename + ".json", "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        return loaded_model
    
    def delete_model(self, filename):
        shutil.rmtree(self.directory + "\\" + filename)
        print("Deleting: "+filename)
    
    @staticmethod
    def _generate_model_filename(model):
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        name = r"model_" + dt_string
        return name
    
    def _get_all_subdirs(self):
        subdirs = [d for d in os.listdir(self.directory) if os.path.isdir(os.path.join(self.directory,d))]
        return subdirs