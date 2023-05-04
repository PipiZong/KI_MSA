import config_adapter as config
from train.fusion_adapter_ft_supcl import TVA_train_fusion, TVA_test_fusion
import datetime
from util.common import set_random_seed

if __name__ == '__main__':

    load_metric = config.MOSI.downStream.load_metric
    check_list = config.MOSI.downStream.check_list
    metric = config.MOSI.downStream.metric
    check = config.MOSI.downStream.check
    result_path = config.MOSI.path.result_path
    seed = config.seed
    result_M = {}
    for s in seed:
        config.seed = s
        config.MOSI.path.encoder_path = config.MOSI.path.model_path + str(s) + '/'

        set_random_seed(s)
        print('TVA_fusion_adapter')
        TVA_train_fusion('TVA_fusion_adapterMAE', check=check, config=config)
        result_M[s] = TVA_test_fusion('TVA_fusion_adapterMAE', check_list=check_list, config=config)

    result_path = result_path + datetime.datetime.now().strftime('%Y-%m-%d-%H%M') + '-fusion' + '.csv'
    with open(result_path, 'w') as file:
        for index, result in enumerate([
            result_M,
        ]):
            file.write('Method%s\n' % index)
            file.write('seed,Has0_acc_2,Has0_F1_score,Non0_acc_2,Non0_F1_score,Mult_acc_5,Mult_acc_7,MAE,Corr\n')
            for s in seed:
                log = '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (s,
                                                        result[s][metric]['Has0_acc_2'],
                                                        result[s][metric]['Has0_F1_score'],
                                                        result[s][metric]['Non0_acc_2'],
                                                        result[s][metric]['Non0_F1_score'],
                                                        result[s][metric]['Mult_acc_5'],
                                                        result[s][metric]['Mult_acc_7'],
                                                        result[s][metric]['MAE'],
                                                        result[s][metric]['Corr'])
                file.write(log)
