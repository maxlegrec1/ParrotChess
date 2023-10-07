

def parameters() -> dict:
    return {
        "training_args" :  {"num_step": 1000},
        
        "shared_parameters" : {"batch_size" : 32, "path_pgn": "human.pgn"},
        
        "model" : 'ResNet',
        
        "trainer" : "default_trainer",
        
        "data_generator": "human_generator",
        
        
        
        
        
        
    }