## Neural Kalman Filtering for Robust Temporal Recommendation

Hi there👋

This is the official implementation of **NeuFilter**, which is accepted by WSDM 2024.

We hope this code helps you well. If you use this code in your work, please cite our paper.





#### How to run this code

##### Step 1: Check the compatibility of your running environment. Generally, different running environments will still have a chance to cause different experimental results though all random processes are fixed in the code. Our running environment is

```
- GPU:  NVIDIA GeForce RTX 3080 (11GB)
- CUDA: 11.6
- Python Packages:
  - numpy: 1.23.5
  - pandas: 1.5.3
  - python: 3.8.16
  - pytorch: 1.13.0
  - scipy: 1.10.1
```



##### Step 2: Prepare the datasets. Please put your datasets under the folder ```data/``` in the corresponding tasks.

We only provide the smallest dataset ```video``` for the ```item recommendation``` task due to the upload speed issue. Other datasets can be accessed from the following links:

* ```item recommendation``` task
  * ML100K, Yoochoosebuy, ML1M: https://github.com/FDUDSDE/CoPE

* ```future interaction prediction ``` task and ```state change prediction``` task
  * Wikipedia: http://snap.stanford.edu/jodie/wikipedia.csv
  * Reddit: http://snap.stanford.edu/jodie/reddit.csv
  * LastFM: http://snap.stanford.edu/jodie/lastfm.csv

> Note that if you use your own datasets, please check their format so as to make sure that it matches the input format of ```NeuFilter```. The dataset format can refer to https://github.com/claws-lab/jodie.



##### Step 3: Run the code. 

* For ```item recommendation``` task, please use the following commands:

  ```bash
  cd item_recommendation
  mkdir saved_models  # If this directory exists, omit this code.
  python main.py --network video --gpu <your_gpu_index>
  ```



* For ```future interaction prediction``` task, please use the following commands

  ```bash
  # For Wikipedia dataset
  cd future_interaction_prediction
  mkdir saved_models  # If this directory exists, omit this code.
  python main.py --network wikipedia --gpu <your_gpu_index> --embedding_dim 16 --num_layer 4 --reg_factor1 1.0 --reg_factor2 0.003 --lr 0.001
  
  # For LastFM dataset
  cd future_interaction_prediction
  mkdir saved_models  # If this directory exists, omit this code.
  python main.py --network lastfm --gpu <your_gpu_index>
  ```



* For ```state change prediction``` task, please use the following commands:

  ```bash
  cd state_change_prediction
  mkdir saved_models  # If this directory exists, omit this code.
  python main.py --network wikipedia --gpu <your_gpu_index>
  ```