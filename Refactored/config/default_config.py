

def parameters() -> dict:
    return {
        "training_args" :  {"num_step": 1000, 'warmup_steps':2},
        
        "shared_parameters" : {"batch_size" : 256, 'lr_start': 1e-4, 'lr':1e-4, 'num_filters': 512, 'num_residuals': 6, 
                               "mini_res_channels" : 64,
                               "path_pgn": "../drive/MyDrive/human.pgn",
                               "num_iters":5},
        
        "model" : 'MiniResidual_TC',
        
        "trainer" : "default_trainer",
        
        "data_generator": "gen_TC_par",
        
        
        
        
        
        
    }
