

def parameters() -> dict:
    return {
        "training_args" :  {"num_step": 1000, 'warmup_steps':2000, "num_report_steps": 1000, "resume_id":None, "start_from": 0, "total_num_steps": 10000000},
        
        "shared_parameters" : {"batch_size" : 256, 'lr_start': 1e-20 ,'lr':1e-20, "path_pgn": "pros.pgn",'num_channels':102,'num_filters':1024,'num_residuals':6,'mini_res_channels':64},
        
        "model" : 'BT4_LoRA',
        
        "trainer" : "default_trainer",
        
        "data_generator": "gen_TC_value",    
    }
