

########### MODELS
import models.Leela_ResNet as model
import models.MiniResidualEmbedding as MiniResidual
import models.MiniResidualEmbedding_TC as MiniResidual_TC
import models.MiniResTCEvaluater as MiniResTCEvaluater
########### TRAINERS
import trainers.default_trainer as default_trainer
import trainers.trainer_evaluater as trainer_evaluater

########### DATA GEN

import data_gen.default_gen as default_gen
import data_gen.gen_castling as gen_castling
import data_gen.uniform_gen as uniform_gen
import data_gen.gen_TC as gen_TC
import data_gen.gen_TC_thinker as gen_TC_thinker
import data_gen.gen_TC_par as gen_TC_par
def from_string_to_fun():
    
    return {
        
        
        
        "ResNet": model.create_model,
        
        "MiniResidual": MiniResidual.create_model,

        "MiniResidual_TC": MiniResidual_TC.create_model,

        "MiniResTCEvaluater": MiniResTCEvaluater.create_model,
        
        "default_trainer": default_trainer.trainer,
        
        "trainer_evaluater": trainer_evaluater.trainer,
        
        "human_generator": default_gen.data_gen,
        
        "castling_generator": gen_castling.data_gen,
        
        "uniform_generator" : uniform_gen.data_gen,
        
        "TC_generator" : gen_TC.data_gen,

        "TC_thinker" : gen_TC_thinker.data_gen,

        "gen_TC" : gen_TC.data_gen,
    }
