# The Causal Impact of Credit Lines on Spending Distributions
### 38th Annual AAAI Conference on Artificial Intelligence, Vancouver, Canada.

### Requirements
Python == 3.8.   
Pytorch: 1.8.1+cu102, Sklearn:0.23.2, Numpy: 1.19.2, Pandas: 1.1.3, Matplotlib: 3.3.2   
All the codes are run on GPUs by default. 

### Simulation Experiments
Train the Simulation dataset
```
python3 Simulation.py --start_exp 0 --end_exp 50
```

### Results 
After training, you can run "concat.py" to get the all experiment results.  

### Citation
@inproceedings{li2024causal,
  title={The causal impact of credit lines on spending distributions},
  author={Li, Yijun and Leung, Cheuk Hang and Sun, Xiangqian and Wang, Chaoqun and Huang, Yiyan and Yan, Xing and Wu, Qi and Wang, Dongdong and Huang, Zhixiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={1},
  pages={180--187},
  year={2024}
}
