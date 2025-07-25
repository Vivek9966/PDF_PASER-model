from pathlib import Path
def get_config():
    return {
        "batch_size":4,
        "num_epochs":50,
        "lr" : 10**-4,
        "seq_len":350,
        "d_model":512,
        "lang_src":"en",
        "lang_tgt":"fr",
        "model_folder":"weights",
        "model_basename":"multi_head_transformer",
        "preload":None,
        "tokenizer_file":f"tokenizer_en_it.json",
        "experiment_name":"runs/model"


    }
def get_weights_file_path(config,epoch):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)

