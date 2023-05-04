import torch
import config_adapter as default_config
from model.TVA_fusion_adapter_supcl import TVA_fusion
from dataloader.MOSI import MOSIDataloader
from tqdm import tqdm
import transformers as trans
from util.metrics import Metrics
from util.common import write_log, check_and_save
import datetime


def TVA_train_fusion(name, check=None, model_type='all', config=default_config):
    print('---------------TVA_Adapter_FT_EXP_%s---------------' % model_type)
    if check is None:
        check = {'Loss': 10000, 'MAE': 100}
    else:
        check = check.copy()
    log_path = config.LOGPATH + "MOSI_TVA_adapter_experiment." + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H%M') + '.log'
    metrics = Metrics()

    if not isinstance(config.Adapter.adapter_list_t, list):
        config.Adapter.adapter_list_t = config.Adapter.adapter_list_t.split(',')
        config.Adapter.adapter_list_t = [int(i) for i in config.Adapter.adapter_list_t]
        config.Adapter.adapter_list_v = config.Adapter.adapter_list_v.split(',')
        config.Adapter.adapter_list_v = [int(i) for i in config.Adapter.adapter_list_v]
        config.Adapter.adapter_list_a = config.Adapter.adapter_list_a.split(',')
        config.Adapter.adapter_list_a = [int(i) for i in config.Adapter.adapter_list_a]


    model = TVA_fusion(config=config)
    model_dict = model.state_dict()
    pretrained_model_path = model.model_path + name + '.ckpt'
    pretrained_model_dict = torch.load(pretrained_model_path, map_location=model.device)
    # print('pretrained model state dict is: %s' % pretrained_model_dict.keys())
    # for item in ['com_dense', 'fusion_dropout', '_decoder']:
    #     if item in pretrained_model_dict:
    #         pretrained_model_dict.pop(item) ###TODO:classifier params can be initialed
    pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}
    model_dict.update(pretrained_model_dict)
    model.load_state_dict(model_dict)

    #### freeze adapters, fine tune others
    train_bool = [True, False, True, True]
    model.set_train(train_bool)
    for n, p in model.named_parameters():
        for item in ['adapter_', 'pooler_t', 'ln_']:
            if item in n:
                p.requires_grad = False


    device = config.DEVICE
    batch_size = config.MOSI.downStream.TVAExp_fusion.batch_size
    total_epoch = config.MOSI.downStream.TVAExp_fusion.epoch
    num_warm_up = config.MOSI.downStream.TVAExp_fusion.num_warm_up

    train_data = MOSIDataloader('train', batch_size=batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    t_encoder_params = list(model.text_encoder.named_parameters())
    a_encoder_params = list(model.audio_encoder.named_parameters())
    v_encoder_params = list(model.vision_encoder.named_parameters())
    t_params_decay = [p for n, p in t_encoder_params if not any(nd in n for nd in no_decay)]
    t_params_no_decay = [p for n, p in t_encoder_params if any(nd in n for nd in no_decay)]
    a_params = [p for n, p in a_encoder_params]
    v_params = [p for n, p in v_encoder_params]
    model_params_other = [p for n, p in list(model.named_parameters()) if 'text_encoder' not in n and \
                          'audio_encoder' not in n and 'vision_encoder' not in n]

    optimizer_grouped_parameters = [
        {'params': t_params_decay, 'weight_decay': config.MOSI.downStream.TVAExp_fusion.t_decay,
         'lr': config.MOSI.downStream.TVAExp_fusion.t_lr},
        {'params': t_params_no_decay, 'weight_decay': 0.0, 'lr': config.MOSI.downStream.TVAExp_fusion.t_lr},
        {'params': a_params, 'weight_decay': config.MOSI.downStream.TVAExp_fusion.a_decay,
         'lr': config.MOSI.downStream.TVAExp_fusion.a_lr},
        {'params': v_params, 'weight_decay': config.MOSI.downStream.TVAExp_fusion.v_decay,
         'lr': config.MOSI.downStream.TVAExp_fusion.v_lr},
        {'params': model_params_other, 'weight_decay': config.MOSI.downStream.TVAExp_fusion.other_decay,
         'lr': config.MOSI.downStream.TVAExp_fusion.other_lr}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    scheduler = trans.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        num_warm_up * (len(train_data))), num_training_steps=total_epoch * len(train_data))
    model.to(device)

    loss_m = 0
    all_loss = 0

    save_start_epoch = 1 ###model.zero_grad()
    for epoch in range(1, total_epoch + 1):

        train_loss = 0
        pred_loss = 0
        model.train()

        bar = tqdm(train_data, disable=False)
        for index, sample1 in enumerate(bar):
            try:
                bar.set_description("Epoch:%d|All_loss:%s|Pred_Loss:%s" % (
                    epoch, all_loss.item(), loss_m.item()))
            except:
                bar.set_description(
                    "Epoch:%d|All_loss:%s|Pred_Loss:%s" % (epoch, all_loss, loss_m))

            optimizer.zero_grad()

            pred, fea, all_loss, loss_m = model(sample1, return_loss=True)

            all_loss.backward()
            train_loss += all_loss.mean().item()
            pred_loss += loss_m.mean().item()

            optimizer.step()
            scheduler.step()
        print('Epoch %s Train loss: %.4f, Pred loss: %.4f' % (epoch, train_loss, pred_loss))
        print("EVAL valid")
        result, result_loss = eval(model, metrics, 'valid', device, config)
        log = 'TVA_%s_ValidAcc\n\tEpoch:%d\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
              'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
              '------------------------------------------' % (model_type,
                                                              epoch, result['Has0_acc_2'],
                                                              result['Has0_F1_score'],
                                                              result['Non0_acc_2'], result['Non0_F1_score'],
                                                              result['Mult_acc_5'],
                                                              result['Mult_acc_7'], result['MAE'],
                                                              result['Corr'],
                                                              result_loss)
        print(log)
        write_log(log, path=log_path)

        if epoch > save_start_epoch:
            check = check_and_save(epoch, model, result, check, save_model=True, name=name)


def eval(model, metrics=None, eval_data=None, device=None, config=default_config):
    with torch.no_grad():
        model.eval()
        if device is None: device = config.DEVICE
        if eval_data is None:
            eval_data = MOSIDataloader('test', shuffle=False, num_workers=0,
                                       batch_size=config.MOSI.downStream.TVAExp_fusion.batch_size)
        else:
            eval_data = MOSIDataloader(eval_data, shuffle=False, num_workers=0,
                                       batch_size=config.MOSI.downStream.TVAExp_fusion.batch_size)
        if metrics is None: metrics = Metrics()
        pred = []
        truth = []
        loss = 0
        bar = tqdm(eval_data, disable=True)
        for index, sample in enumerate(bar):
            label = sample['regression_labels'].clone().detach().to(device).float()
            _pred, fea, _all_loss, _loss= model(sample, return_loss=True)
            pred.append(_pred.view(-1))
            truth.append(label)
            loss += _loss.mean().item()
        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu'))
        eval_results = metrics.eval_mosei_regression(truth, pred)
        eval_results['Loss'] = loss
        model.train()
    return eval_results, loss


def TVA_test_fusion(name, check_list=None, model_type='all', config=default_config):
    if check_list is None: check_list = ['Has0_F1_score', 'MAE']
    if not isinstance(check_list, list): check_list = [check_list]
    seed = config.seed
    log_path = config.LOGPATH + "MOSI_TVA_adapter_experiment_Test." + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H%M') + '_seed_' + str(seed) + '.log'


    model = TVA_fusion(config=config)

    device = config.DEVICE
    model.to(device)
    check = {}
    result = None
    print('Evaluating model:' + model_type)
    for metric in check_list:
        print('Result for best ' + metric)
        model.load_model(name=name + metric, load_pretrain=True)
        result, loss = eval(model=model, device=device, config=config)
        check[metric] = {}
        for key in result.keys():
            check[metric][key] = result[key]

        log = 'TVA_%s_TestAcc\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
              'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
              '------------------------------------------' % (model_type,
                                                              result['Has0_acc_2'], result['Has0_F1_score'],
                                                              result['Non0_acc_2'], result['Non0_F1_score'],
                                                              result['Mult_acc_5'],
                                                              result['Mult_acc_7'], result['MAE'], result['Corr'], loss)

        print(log)
        write_log(metric + '\n' + log, log_path)

    return check
