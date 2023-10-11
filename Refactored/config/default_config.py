

def parameters() -> dict:
    return {
        "training_args" :  {"num_step": 1000, 'warmup_steps':2},
        
        "shared_parameters" : {"batch_size" : 256, 'lr_start': 1e-4, 'lr':1e-4, 'num_filters': 512, 'num_residuals': 12, 
                               "mini_res_channels" : 64,
                               "path_pgn": "human.pgn"},
        
        "model" : 'MiniResidual',
        
        "trainer" : "default_trainer",
        
        "data_generator": "castling_generator",
        
        
        
        
        
        
    }