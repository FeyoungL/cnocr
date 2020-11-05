DATA_ROOT_DIR = data/sample-data
REC_DATA_ROOT_DIR = data/sample-data-lst

# `EMB_MODEL_TYPE` 可取值：['conv', 'conv-lite-rnn', 'densenet', 'densenet-lite']
EMB_MODEL_TYPE = densenet-lite
# `SEQ_MODEL_TYPE` 可取值：['lstm', 'gru', 'fc']
SEQ_MODEL_TYPE = gru
MODEL_NAME = $(EMB_MODEL_TYPE)-$(SEQ_MODEL_TYPE)

# 产生 *.lst 文件
gen-lst:
	python scripts/im2rec.py --list --num-label 20 --chunks 1 \
		--train-idx-fp $(DATA_ROOT_DIR)/train.txt --test-idx-fp $(DATA_ROOT_DIR)/test.txt --prefix $(REC_DATA_ROOT_DIR)/sample-data

self-gen-lst:
	python scripts/im2rec.py --list --num-label 20 --chunks 1 \
	--train-idx-fp datasets/self_dataset_001/train2.txt --test-idx-fp datasets/self_dataset_001/test2.txt --prefix $(REC_DATA_ROOT_DIR)/sample-data

# 利用 *.lst 文件产生 *.idx 和 *.rec 文件。
# 真正的图片文件存储在 `examples` 目录，可通过 `--root` 指定。
gen-rec:
	python scripts/im2rec.py --pack-label --color 1 --num-thread 2 --prefix $(REC_DATA_ROOT_DIR) --root examples

self-gen-rec:
	python scripts/im2rec.py --pack-label --color 1 --num-thread 2 --prefix $(REC_DATA_ROOT_DIR) --root datasets/self_dataset_001/

# 训练模型
train:
	python scripts/cnocr_train.py --gpu 0 --emb_model_type $(EMB_MODEL_TYPE) --seq_model_type $(SEQ_MODEL_TYPE) \
		--optimizer adam --epoch 20 --lr 1e-4 \
		--train_file $(REC_DATA_ROOT_DIR)/sample-data_train --test_file $(REC_DATA_ROOT_DIR)/sample-data_test

# 在测试集上评估模型，所有badcases的具体信息会存放到文件夹 `evaluate/$(MODEL_NAME)` 中
evaluate:
	python scripts/cnocr_evaluate.py --model-name $(MODEL_NAME) --model-epoch 18 -v -i $(DATA_ROOT_DIR)/test.txt \
		--image-prefix-dir examples --batch-size 128 -o evaluate/$(MODEL_NAME)

self-evaluate:
	python scripts/cnocr_evaluate.py --model-name $(MODEL_NAME) --model-epoch 18 -v -i $(DATA_ROOT_DIR)/test.txt \
		--image-prefix-dir datasets/self_dataset_001/ --batch-size 128 -o evaluate/$(MODEL_NAME)


predict:
	python scripts/cnocr_predict.py --model_name $(MODEL_NAME) --file examples/rand_cn1.png


package:
	python setup.py sdist bdist_wheel

VERSION = 1.2.1
upload:
	python -m twine upload  dist/cnocr-$(VERSION)* --verbose


.PHONY: gen-lst gen-rec train evaluate predict package upload self-gen-lst self-gen-rec self-evaluate
