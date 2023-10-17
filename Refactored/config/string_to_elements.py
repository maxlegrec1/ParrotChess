

########### MODELS
import models.Leela_ResNet as model
import models.MiniResidualEmbedding as MiniResidual

########### TRAINERS
import trainers.default_trainer as default_trainer


########### DATA GEN

import data_gen.default_gen as default_gen
import data_gen.gen_castling as gen_castling
import data_gen.uniform_gen as uniform_gen
def from_string_to_fun():
    
    return {
        
        
        
        "ResNet": model.create_model,
        
        "MiniResidual": MiniResidual.create_model,
        
        "default_trainer": default_trainer.trainer,
        
        "human_generator": default_gen.data_gen,
        
        "castling_generator": gen_castling.data_gen,
        
        "uniform_generator" : uniform_gen.data_gen,
    }