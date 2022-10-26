# ASBP
This code is the source code of our paper "An Angular Shrinkage BERT model for Few-shot Relation Extraction
with None-of-the-above Detection".

Our code is based on [Bert-Pair](https://github.com/thunlp/fewrel).


# Data
For the Details of training data, you  can refer to [FewRel](https://thunlp.github.io/2/fewrel2_da.html).

## Re-split the dataset
We follow [ConceptFERE](https://github.com/LittleGuoKe/ConceptFERE) to divide the original training dataset into a new training dataset and a new validation dataset, and the validation set in the original dataset is used as the new test dataset. You can find the corresponding code in https://github.com/LittleGuoKe/ConceptFERE/blob/main/fewshot_re_kit/utils.py 
## Randomness
Due to the randomness of the experiments of the FSRE task, the results in our paper are the average of the results of 5 runs

# Pretrained Model
You can get a PyTorch version of BERT model from [huggingface](https://huggingface.co/bert-base-uncased).
# Training

You can use demo.py for training and testing:

```
python demo.py  --trainN 5 --N 5 --K 1 --Q 1 --model pair --encoder bert --pair --hidden_size 768 --val_step 500 --batch_size 8 --fp16 --na_rate 5 --val_iter 500
```

--trainN: N in train.
 
--N  --K: N-way-K-shot.

--Q: Num of query per class.

--model: specify the name of the model, such as proto, pair, etc.

--val_step: val after training how many iters.

--val_iter: num of iters in validation.

--fp16: use nvidia apex fp16.

--na_rate NOTA rate(1 for 15%, 5 for 50%)

--only_test: add to test a trained model.
