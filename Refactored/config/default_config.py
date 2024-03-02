

def parameters() -> dict:
    return {
        "training_args" :  {"num_step": 1000, 'warmup_steps':2000, "num_report_steps": 1000, "resume_id":"9puxko1e", "start_from": 1000, "total_num_steps": 10000000},
        
        "shared_parameters" : {"batch_size" : 256, 'lr_start': 1e-5 ,'lr':5e-5, "path_pgn": "human2.pgn"},
        
        "model" : 'BT4',
        
        "trainer" : "default_trainer",
        
        "data_generator": "gen_TC",    
    }
