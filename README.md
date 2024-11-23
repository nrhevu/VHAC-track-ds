# VHAC-track-ds

## Environments
Create environments with conda: `conda env create -f environment.yml`


## Project Structure

Make sure to download all file and map to this project structure to reproduce the result

```shell
.
├── SELFRec
├── experimentals
├── recvae
├── data
│   ├── data_final.csv
│   ├── public_testset.csv
│   ├── test.csv
│   ├── test.txt
│   ├── test_set_private.csv
│   ├── train.csv
│   ├── train.txt
│   └── training_set.csv
├── runs
│   ├── ALS
│   │   ├── csr_train.pkl
│   │   ├── item_embedding.pkl
│   │   ├── itemmap.pkl
│   │   ├── model.pkl
│   │   ├── user_embedding.pkl
│   │   └── usermap.pkl
│   ├── DirectAU
│   │   ├── item_embedding.pkl
│   │   ├── model.pkl
│   │   ├── predict.csv
│   │   ├── predict_directau_final_data.zip
│   │   └── user_embedding.pkl
│   ├── LightGCN
│   │   ├── new
│   │   │   ├── directau_private_item_embedding.pkl
│   │   │   └── directau_private_user_embedding.pkl
│   │   ├── item_embedding.pkl
│   │   ├── model.pkl
│   │   ├── predict.csv
│   │   ├── predict_lightgcn_final_data.zip
│   │   ├── user_clusters.csv
│   │   └── user_embedding.pkl
│   ├── RecVAE
│   │   ├── data.csv
│   │   ├── model.pt
│   │   ├── predict.csv
│   │   ├── result_csp.pkl
│   │   ├── test_te.csv
│   │   ├── test_tr.csv
│   │   ├── testset_recvae.csv
│   │   ├── train.csv
│   │   ├── unique_sid.txt
│   │   ├── unique_uid.txt
│   │   ├── validation_te.csv
│   │   └── validation_tr.csv
│   ├── SAR
│   │   └── model.pkl
│   ├── SimGCL
│   │   ├── item_embedding.pkl
│   │   ├── model.pkl
│   │   ├── predict.csv
│   │   ├── predict_simgcl_final_data.zip
│   │   └── user_embedding.pkl
│   ├── XSimGCL
│   │   ├── item_embedding.pickle
│   │   ├── model.pkl
│   │   └── user_embedding.pickle
│   ├── co-visitation-matrix
│   │   ├── buy_order_denoise.csv
│   │   ├── buy_order_full.csv
│   │   ├── buy_order_standard.csv
│   │   ├── cart_order_denoise.csv
│   │   ├── cart_order_full.csv
│   │   ├── cart_order_standard.csv
│   │   ├── click_order_action_num_reverse_denoise.csv
│   │   ├── click_order_action_num_reverse_full.csv
│   │   ├── click_order_action_num_reverse_standard.csv
│   │   ├── click_order_log_recency_score_denoise.csv
│   │   ├── click_order_log_recency_score_full.csv
│   │   ├── click_order_log_recency_score_standard.csv
│   │   ├── click_order_type_weighted_log_recency_score_denoise.csv
│   │   ├── click_order_type_weighted_log_recency_score_full.csv
│   │   ├── click_order_type_weighted_log_recency_score_standard.csv
│   │   └── predict.csv
│   ├── private-test-attempt
│   │   ├── ALS_new_cluster_top10.csv
│   │   ├── DirectAU_predict.csv
│   │   ├── LightGCN_new_cluster_top10.csv
│   │   ├── RecVAE_05981.csv
│   │   ├── SAR_std_data.csv
│   │   ├── SimGCL_predict.csv
│   │   ├── XSimFull_05855.csv
│   │   ├── predict_ALS.csv
│   │   ├── predict_XSim_denoise_05775.csv
│   │   ├── predict_XSim_ensemble_noscore.csv
│   │   ├── predict_XSim_standard_05772.csv
│   │   ├── predict_ensemble_8file.csv
│   │   └── predict_full_rerank.csv
│   ├── private-test-attempt-final
│   └── reranking
│       ├── temp_csv
│       ├── input.csv
│       ├── input_submit.csv
│       ├── output.csv
│       ├── output_submit.csv
│       └── predict.csv
├── submission
├── weight-model
│   ├── emb_item_XSim128.pickle
│   ├── emb_item_XSim_denoise.pickle
│   ├── emb_item_XSim_full.pickle
│   ├── emb_item_XSim_multiverse.pickle
│   ├── emb_item_XSim_standard.pickle
│   ├── emb_user_XSim_128.pickle
│   ├── emb_user_XSim_denoise.pickle
│   ├── emb_user_XSim_full.pickle
│   ├── emb_user_XSim_multiverse.pickle
│   ├── emb_user_XSim_standard.pickle
│   ├── item_embedding_lightgcn.pkl
│   ├── item_embedding_recvae.pkl
│   ├── item_embedding_simgcl.pkl
│   ├── rec_list.pickle
│   ├── user_embedding_lightgcn.pkl
│   ├── user_embedding_recvae.pkl
│   └── user_embedding_simgcl.pkl
├── ALS-Training.ipynb
├── Co-Visitation-Matrix.ipynb
├── Pipeline-Road-to-The-Championship.ipynb
├── README.md
├── RecVAE-Training.ipynb
├── Rerank.ipynb
├── SAR-Training.ipynb
├── SELFRec-Training.ipynb
└── environment.yml
```

## Retrain model
To retrain RecVAE, check the notebook `RecVAE-Training.ipynb`

To retrain SELFRec models, which includes XSimGCL, LightGCN, DirectAU, SimGCL, check the notebook `SELFRec-Training.iypnb`

To retrain ALS, check the notebook `ALS-Training.ipynb`

To retrain SAR, check the notebook `SAR-Training.ipynb`

## Download weight and runs files
`data\` can be downloaded [here](https://huggingface.co/datasets/Rhev124/VHAC-track-ds)

`runs\` can be downloaded [here](https://huggingface.co/datasets/Rhev124/VHAC-track-ds-runs)

`weight-model\` can be downloaded [here](https://huggingface.co/datasets/Rhev124/VHAC-track-ds-weight-model)

## Reproduce The Result
To reproduce the result: Check the notebook `Pipeline-Road-to-The-Championship.ipynb`. Run the Ensemble Final session to reproduce final submission.

## More
Our experiments are stored in `experimentals` path. Check out if you are interested.
