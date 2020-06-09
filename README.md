### Running Requirements and Dependencies

* Single NVIDIA GPU and at least 60 GB memory
* TensorFlow 2.2
* Transformers



### Code Structure

 ```
cs263-final-project
  |__
     |__ codes
        |__ __init__.py
        |__ run.py
        |__ data_process.py
        |__ roberta_mc.py
        |__ roberta_mc_cls.py
        |__ roberta_mc_all_token.py
     |__ README.md
 ```



### Running Commands

Download the Data-set:

```shell
wget https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip
unzip winogrande_1.1.zip
mv winogrande_1.1 data
rm winogrande_1.1.zip
```

Using the CLS token for classification:
```shell
cp codes/roberta_mc_cls.py codes/roberta_mc.py
```

Using all tokens for classfication:
```shell
cp codes/roberta_mc_all_token.py codes/roberta_mc.py
```

Train and evaluate:

```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)

python3 codes/run.py --model_name roberta-large --train_batch_size 64 --learning_rate 1e-5 --num_train_epochs 3 --output_dir ./output/models --train --eval --overwrite_output_dir
```

