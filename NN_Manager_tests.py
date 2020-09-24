# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:51:20 2020

@author: Izabela Perenc
"""

import keras
from   keras import layers
import unittest

from NN_Manager import *

class SaveModelTest(unittest.TestCase):
    
    def setUp(self):
        
        self.models = []
        self.manager = NN_Manager(r".\temp\\")
         
        inputs = keras.Input(shape=(4,))
        l0 = layers.Dense(5, activation="relu")(inputs)
        l1 = layers.Dense(6, activation="relu")(l0)
        l2 = layers.Dense(3, activation="relu")(l1)
        outputs = layers.Dense(2, activation='sigmoid')(l2)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="model")
        name = self.manager.save_model(model, True)
        self.models.append(name)

    def tearDown(self):
        for model in self.models:
            self.manager.delete_model(model)
        
    def test_save_and_load(self):
        
        inputs = keras.Input(shape=(4,))
        l0 = layers.Dense(8, activation="relu")(inputs)
        l1 = layers.Dense(6, activation="relu")(l0)
        l2 = layers.Dense(3, activation="relu")(l1)
        outputs = layers.Dense(2, activation='sigmoid')(l2)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="model")
        name = self.manager.save_model(model)
        loaded_model = self.manager.load_model(name)
        
        self.assertEqual(model.get_config(), loaded_model.get_config())
        
        self.models.append(name)

    def test_save_and_load_while_model_exists_and_its_ok(self):
        
        inputs = keras.Input(shape=(4,))
        l0 = layers.Dense(5, activation="relu")(inputs)
        l1 = layers.Dense(6, activation="relu")(l0)
        l2 = layers.Dense(3, activation="relu")(l1)
        outputs = layers.Dense(2, activation='sigmoid')(l2)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="model")
        name = self.manager.save_model(model, True)
        loaded_model = self.manager.load_model(name)
        
        self.assertEqual(model.get_config(), loaded_model.get_config())
        
        self.models.append(name)
        
    def test_save_and_load_while_model_exists_and_its_not_ok(self):
        
        inputs = keras.Input(shape=(4,))
        l0 = layers.Dense(5, activation="relu")(inputs)
        l1 = layers.Dense(6, activation="relu")(l0)
        l2 = layers.Dense(3, activation="relu")(l1)
        outputs = layers.Dense(2, activation='sigmoid')(l2)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="model")
        name = self.manager.save_model(model) # allow_duplicates set to False
        
        self.assertEqual(name, None)
        
if __name__ == '__main__':
    unittest.main()