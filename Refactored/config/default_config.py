

def parameters() -> dict:
    return {
        "training_args" :  {"num_step": 1000, 'warmup_steps':2000, "num_report_steps": 1000, "resume_id":None, "start_from": 0, "total_num_steps": 10000000},
        
        "shared_parameters" : {"batch_size" : 256, 'lr_start': 1e-5 ,'lr':5e-5, "path_pgn": "/media/maxime/Crucial X8/Gigachad/engine.pgn",'num_channels':102,'num_filters':1024,'num_residuals':6,'mini_res_channels':64},
        
        "model" : 'bt4_moe',
        
        "trainer" : "default_trainer",
        
        "data_generator": "gen_TC",    
    }
