from transformers import GPT2Config

config = GPT2Config()

config.n_embd = 128
config.n_head = 8
config.n_layer = 4
config.vocab_size = 500

config.to_json_file('gpt2-small-config.json')
