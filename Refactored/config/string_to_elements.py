

########### MODELS
import models.BT4 as BT4
import models.BT5 as BT5
import models.BT4_LoRA as BT4_LoRA
import models.bt4_moe as bt4_moe

########### TRAINERS
import trainers.default_trainer as default_trainer


########### DATA GEN

import data_gen.gen_TC as gen_TC
import data_gen.gen_leela as gen_leela
def from_string_to_fun():
    
    return {

        "BT4": BT4.create_model,

        "BT4_LoRA": BT4_LoRA.create_model,

        "BT5": BT5.create_model,

        "bt4_moe": bt4_moe.create_model,

        "default_trainer": default_trainer.trainer,
        
        "TC_generator" : gen_TC.data_gen,

        "gen_TC" : gen_TC.data_gen,

        "gen_leela" : gen_leela.data_gen,
    }
