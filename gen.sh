#python3.6 train.py --data_path STAR/dialogues/ --schema_path STAR/tasks/ --token_vocab_path bert-base-uncased-vocab.txt --output_dir final_domaintransfer_zeroshot_sam_best/ --task action --num_epochs 0 --use_schema
python3.6 train.py --data_path STAR/dialogues/ --schema_path STAR/tasks/ --token_vocab_path bert-base-uncased-vocab.txt --output_dir zeroshot_domaintransfer_copygen_sam/ --action_output_dir final_domaintransfer_zeroshot_sam_best/ --task generation --num_epochs 10 --train_batch_size 4 --grad_accum 8 --use_schema

