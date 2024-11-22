# VHAC-track-ds

## Environments
Create environments with conda: `conda env create -f environment.yml`


## Project Structure

Make sure to download all file and map to this project structure to reproduce the result

```shell
.
├── SELFRec
├── data
│   ├── data_final.csv
│   ├── public_testset.csv
│   ├── test.csv
│   ├── test.txt
│   ├── test_set_private.csv
│   ├── train.csv
│   ├── train.txt
│   └── training_set.csv
├── experimentals
│   ├── Bert4rec.ipynb
│   ├── Conflict_item.ipynb
│   ├── Error Analysis.ipynb
│   ├── Gridsearch.ipynb
│   ├── Multiple View Data.ipynb
│   ├── RecVAE-Get-Embedding.ipynb
│   └── UserCluster.ipynb
├── recvae
│   ├── infer.py
│   ├── latent.py
│   ├── model.py
│   ├── run.py
│   └── utils.py
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
│   │   └── user_embedding.pkl
│   ├── SAR
│   │   └── model.pkl
│   ├── SimGCL
│   │   ├── item_embedding.pkl
│   │   ├── model.pkl
│   │   ├── predict.csv
│   │   ├── predict_simgcl_final_data.zip
│   │   └── user_embedding.pkl
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
│   └── reranking
│       ├── temp_csv
│       ├── input.csv
│       ├── input_submit.csv
│       ├── output.csv
│       ├── output_submit.csv
│       └── predict.csv
├── submission
│   └── predict_ensemble_7file.csv
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
├── RecVAE-Create-Train-Data.ipynb
├── Rerank.ipynb
├── SAR-Training.ipynb
├── SELFRec-Training.ipynb
└── requirements.txt
```

## Retrain model
To retrain RecVAE, check `RecVAE-Training.ipynb`
To retrain SELFRec models, which includes XSimGCL, LightGCN, DirectAU, SimGCL, check `SELFRec-Training.iypnb`
To retrain ALS, check `ALS-Training.ipynb`
To retrain SAR, check `SAR-Training.ipynb`

## Reproduce Result
To reproduce result: Check `Pipeline-Road-to-The-Championship.ipynb`. Run the Ensemble Final session to reproduce final submission.