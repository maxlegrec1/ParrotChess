

########### MODELS
import models.Leela_ResNet as model


########### TRAINERS
import trainers.default_trainer as default_trainer


########### DATA GEN

import data_gen.default_gen as default_gen

def from_string_to_fun():
    
    return {
        
        
        
        "ResNet": model.create_model,
        
        "default_trainer": default_trainer.train,
        
        "human_generator": default_gen.data_gen
    }