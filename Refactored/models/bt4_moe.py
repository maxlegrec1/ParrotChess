import os
import sys

sys.path.append(os.getcwd())
sys.path.append("/media/maxime/Crucial X8/moe/lczero-training")
sys.path.append("/media/maxime/Crucial X8/moe/lczero-training/tf")
sys.path.append("/media/maxime/Crucial X8/moe/lczero-training/tf/proto")



from tf.tfprocess import TFProcess



def create_model(*args,**kwargs):
    import yaml
    with open("/media/maxime/Crucial X8/moe/lczero-training/tf/configs/example.yaml") as file:
        cfg = yaml.safe_load(file)
    tfp = TFProcess(cfg)
    tfp.init_net()

    return tfp.model

#model = create_model()