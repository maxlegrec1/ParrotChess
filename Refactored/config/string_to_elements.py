

########### MODELS
import models.Leela_ResNet as model
import models.MiniResidualEmbedding as MiniResidual
import models.MiniResidualEmbedding_TC as MiniResidual_TC
import models.MiniResTCEvaluater as MiniResTCEvaluater
import models.BT4 as BT4
import models.BT5 as BT5
import models.BT4_LoRA as BT4_LoRA
import models.BT4_LoRA_value as BT4_LoRA_value
import models.ED as ED
import models.bt4_real as bt4_real
########### TRAINERS
import trainers.default_trainer as default_trainer


########### DATA GEN

import data_gen.default_gen as default_gen
import data_gen.gen_castling as gen_castling
import data_gen.uniform_gen as uniform_gen
import data_gen.gen_TC as gen_TC
import data_gen.gen_TC_value as gen_TC_value
import data_gen.gen_TC_thinker as gen_TC_thinker
import data_gen.gen_TC_par as gen_TC_par
def from_string_to_fun():
    
    return {
        
        
        
        "ResNet": model.create_model,
        
        "MiniResidual": MiniResidual.create_model,

        "MiniResidual_TC": MiniResidual_TC.create_model,

        "MiniResTCEvaluater": MiniResTCEvaluater.create_model,

        "BT4": BT4.create_model,

        "bt4_real": bt4_real.create_model,

        "BT4_LoRA": BT4_LoRA.create_model,

        "BT4_LoRA_value": BT4_LoRA_value.create_model,

        "BT5": BT5.create_model,

        "ED": ED.create_model,
        
        "default_trainer": default_trainer.trainer,
        
        "human_generator": default_gen.data_gen,
        
        "castling_generator": gen_castling.data_gen,
        
        "uniform_generator" : uniform_gen.data_gen,
        
        "TC_generator" : gen_TC.data_gen,

        "TC_thinker" : gen_TC_thinker.data_gen,

        "gen_TC_par" : gen_TC_par.data_gen,

        "gen_TC" : gen_TC.data_gen,

        "gen_TC_value" : gen_TC_value.data_gen,
    }
