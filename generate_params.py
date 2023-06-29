import json
import itertools

def generate_param_files(concept_dim_list, max_path_len_list, num_heads_list,
                         max_neighbourhood_sample_list, #sd_of_kernel_init_list,
                         batch_size_list, learning_rate_list, negative_sample_rate_list,
                         with_random_walks_list, multi_task_losses_list, dropout_rate_list,
                         only_hpo_list, l2_strength_list, #warmup_steps_list,
                         input_dropout_rate_list, head_dropout_rate_list):
    param_combinations = itertools.product(
        concept_dim_list, max_path_len_list, num_heads_list,
        max_neighbourhood_sample_list, #sd_of_kernel_init_list,
        batch_size_list, learning_rate_list, negative_sample_rate_list,
        with_random_walks_list, multi_task_losses_list, dropout_rate_list,
        only_hpo_list, l2_strength_list, #warmup_steps_list,
        input_dropout_rate_list, head_dropout_rate_list
    )
    epochs = 150
    early_stopping = 5
    # margin = 1.0
    num_filters = 1024
    compression_ratio = 8
    for i, params in enumerate(param_combinations):

        if params[7]:
            model_name = f"arc_cd{params[0]}_mpl{params[1]}_nh{params[2]}_ms{params[3]}_bs{params[4]}_lr{params[5]}_rw{str(params[7])}_nsr{params[6]}_mtl{params[8][0]}_dor{params[9]}_hpo{params[10]}_l2{params[11]}_idr{params[12]}_hdr{params[13]}"
            param_dict = {
                "model_name": model_name,
                "param_dir": "saved_models/arc",
                "fasttext": "fasttext",
                "figures_dir": "figures",
                "output_dir": "final_model",
                "data_dir": f"stream_data",
                "dict_dir": "dictionaries",
                "epochs": epochs,
                "early_stopping": early_stopping,
                "shuffle": True,
                "repeat": True,

                "concept_dim": params[0],
                "num_filters": num_filters,
                "compression_ratio": compression_ratio,
                "max_path_len": params[1],
                "num_heads": params[2],
                "max_neighbourhood_sample": params[3],
                # "sd_of_kernel_init": params[4],
                "batch_size": params[4],
                "learning_rate": params[5],
                "negative_sample_rate": params[6],
                "with_random_walks": params[7],
                "multi_task_losses": params[8],
                "dropout_rate": params[9],
                "only_hpo": params[10],
                "l2_strength": params[11],
                # "warmup_steps": params[12],

                "input_dropout_rate": params[12],
                "head_dropout_rate": params[13]
            }
        else:
            model_name = f"arc_cd{params[0]}_mpl{params[1]}_nh{params[2]}_ms{params[3]}_bs{params[4]}_lr{params[5]}_rw{params[7]}_dor{params[9]}_hpo{params[10]}_l2{params[11]}_idr{params[12]}_hdr{params[13]}"
            param_dict = {
                "model_name": model_name,
                "param_dir": "saved_models/arc",
                "fasttext": "fasttext",
                "figures_dir": "figures",
                "output_dir": "final_model",
                "data_dir": f"stream_data",
                "dict_dir": "dictionaries",
                "epochs": epochs,
                "early_stopping": early_stopping,
                "shuffle": True,
                "repeat": True,

                "concept_dim": params[0],
                "num_filters": num_filters,
                "compression_ratio": compression_ratio,
                "max_path_len": params[1],
                "num_heads": params[2],
                "max_neighbourhood_sample": params[3],
                # "sd_of_kernel_init": params[4],
                "batch_size": params[4],
                "learning_rate": params[5],
                "negative_sample_rate": params[6],
                "with_random_walks": params[7],
                "dropout_rate": params[9],
                "only_hpo": params[10],
                "l2_strength": params[11],
                # "warmup_steps": params[12],

                "input_dropout_rate": params[12],
                "head_dropout_rate": params[13]
            }

        with open(f"params/{model_name}.json", "w") as f:
            json.dump(param_dict, f)

concept_dim_list = [512, 1024]
max_path_len_list = [3]
num_heads_list = [4]
max_neighbourhood_sample_list = [32]
# sd_of_kernel_init_list = [0.001, 0.01]
batch_size_list = [512]
learning_rate_list = [0.0005]
with_random_walks_list = [True, False]
negative_sample_rate_list = [0.5]
multi_task_losses_list = [[0.5, 0.5]]
dropout_rate_list = [0.1]
only_hpo_list = [True, False]
l2_strength_list = [0, 0.01]
# warmup_steps_list = [500]

input_dropout_rate_list = [0.1]
head_dropout_rate_list = [0.25]

# generate_param_files(concept_dim_list, max_path_len_list, num_heads_list,
#                      max_neighbourhood_sample_list, # sd_of_kernel_init_list,
#                      batch_size_list, learning_rate_list, negative_sample_rate_list,
#                      with_random_walks_list, multi_task_losses_list, dropout_rate_list,
#                      only_hpo_list, l2_strength_list, #warmup_steps_list,
#                      input_dropout_rate_list, head_dropout_rate_list)


import numpy as np

def generate_latex_table(param_combinations):
    # Transposing the parameter combinations to access columns
    param_combinations_T = np.transpose(param_combinations)

    # Identify columns where all values are the same
    mask = [len(set(map(tuple, column) if isinstance(column[0], list) else column)) > 1 for column in param_combinations_T]

    # Apply mask to column names and parameter combinations
    column_names = np.array(["concept_dim", "max_path_len", "num_heads", "max_neighbourhood_sample",
                             "batch_size", "learning_rate", "negative_sample_rate", "with_random_walks",
                             "multi_task_losses", "dropout_rate", "only_hpo", "l2_strength",
                             "input_dropout_rate", "head_dropout_rate"])[mask]

    param_combinations_filtered = list(map(lambda params: np.array(params)[mask], param_combinations))

    # LaTeX table setup
    latex_table = "\\begin{table}[]\n\\begin{tabular}{|" + "c|"*len(column_names) + "}\n\\hline\n"

    # Adding column headers
    latex_table += " & ".join(column_names) + " \\\\ \\hline\n"

    # Adding values for each row
    for params in param_combinations_filtered:
        row_values = " & ".join(map(lambda val: str(val) if not isinstance(val, list) else str(val[0]), params))  # convert each value to string and join them with " & "
        latex_table += row_values + " \\\\ \\hline\n"

    # LaTeX table end
    latex_table += "\\end{tabular}\n\\end{table}"

    return latex_table

param_combinations = list(itertools.product(
    concept_dim_list, max_path_len_list, num_heads_list,
    max_neighbourhood_sample_list, batch_size_list, learning_rate_list,
    negative_sample_rate_list, with_random_walks_list, multi_task_losses_list,
    dropout_rate_list, only_hpo_list, l2_strength_list,
    input_dropout_rate_list, head_dropout_rate_list
))

latex_table = generate_latex_table(param_combinations)

print(latex_table)


