import torch

from .model import RNNSequence, ModelConfig

device = torch.device('cuda')

config = ModelConfig.default()
model = RNNSequence(config)
model.type(torch.float32)
model.to(device)

text_input = torch.full((1, config.short_ctx_len), 3).to(device)

internal = model.input(text_input)
recurrent, internal = model.ponder(model.recurrent_init.clone().unsqueeze(0), internal)
out_token, p_halt = model.decode(recurrent)

print('out', out_token.softmax(-1))
print('p_halt', p_halt)

import time
time.sleep(60)
