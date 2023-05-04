import config_adapter as config
from train.TVA_fusion_train_adapter import TVA_train_fusion
from util.common import set_random_seed

if __name__ == '__main__':
    # follow below performance
    load_metric = config.MOSI.downStream.load_metric
    check_list = config.MOSI.downStream.check_list
    metric = config.MOSI.downStream.metric
    # select which model to save
    check = config.MOSI.downStream.check
    result_path = config.MOSI.path.result_path
    seed = config.seed
    result_M = {}

    config.MOSI.path.encoder_path = config.MOSI.path.model_path + str(seed) + '/'

    set_random_seed(config.seed)
    print('TVA_fusion_adapter')
    TVA_train_fusion('TVA_fusion_adapter', check=check, config=config)

