import torch

def af(adapter_list, adapter_skip_layers, hidden_states, outputs_pretrained, adapter_model):
    hidden_states_last = torch.zeros(outputs_pretrained.size()).to('cuda')
    adapter_hidden_states = []
    adapter_hidden_states_count = 0
    for i, adapter_module in enumerate(adapter_model):
        fusion_state = hidden_states[adapter_list[i]] + hidden_states_last
        hidden_states_last = adapter_module(fusion_state)
        adapter_hidden_states.append(hidden_states_last)
        adapter_hidden_states_count += 1
        if adapter_skip_layers >= 1:  # if adapter_skip_layers>=1, skip connection
            if adapter_hidden_states_count % adapter_skip_layers == 0:
                hidden_states_last = hidden_states_last + adapter_hidden_states[
                    int(adapter_hidden_states_count / adapter_skip_layers)]
    return hidden_states_last