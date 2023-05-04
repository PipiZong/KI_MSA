import torch
from util.common import check_dir

seed = [1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGPATH = 'log/adapter_ft/'
check_dir(LOGPATH)

USEROBERTA = False


class MOSI:
    class path:
        bert_en = 'data/MOSI/bert_en'
        raw_data_path = 'data/MOSI/unaligned_50.pkl'
        raw_data_path_mosei = 'data/MOSEI/unaligned_50.pkl'
        model_path = 'ckpt/encoder'
        if USEROBERTA:
            model_path = model_path + '/roberta/'
        else:
            model_path = model_path + '/bert/'
        check_dir(model_path)
        result_path = 'result/'
        check_dir(result_path)

    class downStream:
        # follow below performance
        metric = 'MAE'
        load_metric = 'best_' + metric
        check_list = [metric]

        # select which model to save
        check = {metric: 10000 if metric == 'Loss' or metric == 'MAE' else 0}

        # parameters
        use_reg = True
        proj_fea_dim = 256
        encoder_fea_dim = 768
        text_fea_dim = 768
        vision_fea_dim = 20
        video_seq_len = 500
        audio_fea_dim = 5
        audio_seq_len = 375
        text_drop_out = 0.5
        vision_drop_out = 0.5
        audio_drop_out = 0.5
        vision_nhead = 8
        audio_nhead = 8
        vision_dim_feedforward = vision_fea_dim
        audio_dim_feedforward = audio_fea_dim
        vision_tf_num_layers = 2
        audio_tf_num_layers = 2
        lamda = 0

        class TVAExp_fusion:
            batch_size = 32
            t_lr = 5e-6
            t_decay = 1e-3
            a_lr = 5e-6
            a_decay = 1e-3
            v_lr = 5e-6
            v_decay = 1e-3
            other_lr = 1e-6
            other_decay = 1e-3

            epoch = 200
            num_warm_up = 10

            post_fusion_dropout = 0.1
            # post_text_dropout = 0.1
            # post_audio_dropout = 0.1
            # post_video_dropout = 0.0
            segating = True

class Adapter:
    output_dir = 'ckpt/adapter/'
    adapter_transformer_layers = 2
    adapter_size = 768
    adapter_list_t = "0,5,10"
    adapter_list_v = "1"
    adapter_list_a = "1"
    adapter_skip_layers = 0
    learning_rate = 5e-5
    weight_decay = 0.0
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    num_train_epochs = 10
    max_steps = 1
    warmup_steps = 1
    save_steps = 1
    eval_steps = 1
    max_save_checkpoints = 100

    project_hidden_size: int = 768
    hidden_act: str = "gelu"
    # adapter_size: int = self.adapter_size  # 64
    adapter_initializer_range: float = 0.0002
    is_decoder: bool = False
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    hidden_size: int = 768
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-05
    max_position_embeddings: int = 512
    num_attention_heads: int = 12
    num_hidden_layers: int = 2
    num_labels: int = 3
    output_attentions: bool = False
    output_hidden_states: bool = False
    # torchscript: bool = False
    type_vocab_size: int = 1
    vocab_size: int = 30522
    chunk_size_feed_forward:int = 0
    add_cross_attention = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

