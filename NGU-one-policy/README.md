# NGU one policy

## Introduction

This folder contains NGU 1 policy code, demo, trained model. 

<p align="center">
  <img src="demo/gif/8-4.gif" width="200">
  <img src="demo/gif/8-4-100.gif" width="200"><br/>
  <i>NGU 1 policy Results</i>
</p>

## NGU hyperparameters

### How to find hyperparameters

First I try a similar policy to pseudo-counts, I use the same hyperparameter set and complete stage 8-4. For rnd, I normalize same as paper:
- normalize rnd = 1 + (rnd - rms_mean) / rms_std
- clip normalize rnd --> [1, 5]
- intrinsic reward = episodic reward (pseudo-counts reward) * clip normalize rnd

Then I remembered that I deducted more reward for dying in stage 8-4 in a part that needed exploring (-100). I changed it to -50 as usual. The algorithm completed this stage with entropy_coef=0.01 (completed in 1/2 test run) but couldn't complete it with entropy_coef = 0.05 (0/1 compelted)! This may be affected by probability since the algorithm is not guaranteed to run well every time (random seed). However, it also shows that a change in the reward system can affect the algorithm's performance and change the correlation between hyperparameters and the environment!

I continued to experiment with the UVFA framework like the paper (training different policies for each actor). Inspired by NGU, my PPO-NGU uses a model with 2 additional inputs, gamma and beta, depending on the actor. Different (gamma, beta) pairs will give different policies. Each actor will have its own (gamma, beta) pair, so it will create a separate policy for that actor. Note that the original NGU uses a more complex model because they add the intrinsic reward directly to the extrinsic reward. I just tried the simple way of adding 2 parameters beta, gamma and it works fine!

I copy the betas, gammas settings from paper. 

This NGU can complete all stages in the first run without hyperparameter tuning. I ran stages 8-4 one more time, the second time I got ~340-350/380-390 rewards (almost completed) and I stopped after 6e6 steps because it took too long (I think this run would have completed with extra running if I wasn't too unlucky).

#### Hyperparameters 1 policy stage 8-4

Below is a detailed hyperparameter table:

| Hyperparameters | Value |
| :--- | :--- |
| **num_envs** | 32 |
| **learn_step** | 512 |
| **batchsize** | 256 |
| **epoch** | 10 |
| **lambda** | 0.95 |
| **gamma** | 0.99 |
| **gamma_int** | 0.99 |
| **learning_rate** | 7e-5 |
| **target_kl** | 0.05 |
| **clip_param** | 0.2 |
| **max_grad_norm** | 0.5 |
| **update_proportion** | 0.1 |
| **norm_adv** | FALSE |
| **int_adv_coef** | 0.5 |
| **ext_adv_coef** | 1 |
| **V_coef** | 0.5 |
| **entropy_coef** | 0.01 |
| **loss_type** | huber |
| **k** | 10 |
| **kernel_cluster_distance** | 0.008 |
| **kernel_epsilon** | 0.0001|
| **c** | 0.001 |
| **sm** | 8 |
| **training_step** | 1196031 | 
| **training_time** | 1 day, 16:17:58 |

#### Detail:
- The hyperparameters for calculating episodic_reward (pseudo-counts reward) are `k = 10, kernel_cluster_distance = 0.008, kernel_epsilon = 0.0001, c = 0.001, and sm = 8`, the same as in the NGU paper.
- `num_envs = 32`, the same as the NGU paper and previous projects.
- `update_proportion = 0.1`, like RND. Additionally, NGU uses 5/80 frames (the last 5 frames in the 80-frame sequence) to train RND and embedding the model, so update_proportion = 6.25% (I rounded it to 0.1).
- `int_adv_coef, int_adv_coef: 0.5 and 1`, like old projects.
- `gamma, gamma_int: 0.99 and 0.99`, like old projects.
- `entropy_coef = 0.01`: I found 0.01 performed better than 0.05 with the NGU PPO.
- `learn_step = 512, batchsize = 256, lambda = 0.95, epoch = 10, lr = 7e-5, target_kl = 0.05, clip_param = 0.2, max_grad_norm = 0.5, norm_adv = false, V_coef = 0.5`, as in previous projects.