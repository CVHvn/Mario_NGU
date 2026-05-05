# NGU

## Introduction

This folder contains NGU code, demo, trained model. 

<p align="center">
  <img src="demo/gif/1-1.gif" width="200">
  <img src="demo/gif/1-2.gif" width="200">
  <img src="demo/gif/1-3.gif" width="200">
  <img src="demo/gif/1-4.gif" width="200"><br/>
  <img src="demo/gif/2-1.gif" width="200">
  <img src="demo/gif/2-2.gif" width="200">
  <img src="demo/gif/2-3.gif" width="200">
  <img src="demo/gif/2-4.gif" width="200"><br/>
  <img src="demo/gif/3-1.gif" width="200">
  <img src="demo/gif/3-2.gif" width="200">
  <img src="demo/gif/3-3.gif" width="200">
  <img src="demo/gif/3-4.gif" width="200"><br/>
  <img src="demo/gif/4-1.gif" width="200">
  <img src="demo/gif/4-2.gif" width="200">
  <img src="demo/gif/4-3.gif" width="200">
  <img src="demo/gif/4-4.gif" width="200"><br/>
  <img src="demo/gif/5-1.gif" width="200">
  <img src="demo/gif/5-2.gif" width="200">
  <img src="demo/gif/5-3.gif" width="200">
  <img src="demo/gif/5-4.gif" width="200"><br/>
  <img src="demo/gif/6-1.gif" width="200">
  <img src="demo/gif/6-2.gif" width="200">
  <img src="demo/gif/6-3.gif" width="200">
  <img src="demo/gif/6-4.gif" width="200"><br/>
  <img src="demo/gif/7-1.gif" width="200">
  <img src="demo/gif/7-2.gif" width="200">
  <img src="demo/gif/7-3.gif" width="200">
  <img src="demo/gif/7-4.gif" width="200"><br/>
  <img src="demo/gif/8-1.gif" width="200">
  <img src="demo/gif/8-2.gif" width="200">
  <img src="demo/gif/8-3.gif" width="200">
  <img src="demo/gif/8-4.gif" width="200"><br/>
  <i>NGU Results</i>
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

### Hyperparameters for all stages

Below is a detailed hyperparameter table for full NGU. This will work for all stages.

| Hyperparameters | Value |
| :--- | :--- |
| **num_envs** | 32 |
| **learn_step** | 512 |
| **batchsize** | 256 |
| **epoch** | 10 |
| **lambda** | 0.95 |
| **gamma_min** | 0.99 |
| **gamma_max** | 0.997 |
| **learning_rate** | 7e-5 |
| **target_kl** | 0.05 |
| **clip_param** | 0.2 |
| **max_grad_norm** | 0.5 |
| **update_proportion** | 0.1 |
| **norm_adv** | FALSE |
| **beta_min** | 0 |
| **beta_max** | 1 |
| **V_coef** | 0.5 |
| **entropy_coef** | 0.01 |
| **loss_type** | huber |
| **k** | 10 |
| **kernel_cluster_distance** | 0.008 |
| **kernel_epsilon** | 0.0001|
| **c** | 0.001 |
| **sm** | 8 |

#### Detail:
- The hyperparameters for calculating episodic_reward (pseudo-counts reward) are `k = 10, kernel_cluster_distance = 0.008, kernel_epsilon = 0.0001, c = 0.001, and sm = 8`, the same as in the NGU paper.
- `num_envs = 32`, the same as the NGU paper and previous projects.
- `update_proportion = 0.1`, like RND. Additionally, NGU uses 5/80 frames (the last 5 frames in the 80-frame sequence) to train RND and embedding the model, so update_proportion = 6.25% (I rounded it to 0.1).
- `beta_min, beta_max: 0 and 1`, the same as the NGU paper. `beta` also corresponds to `int_adv_coef` in other projects (I use beta to match the NGU paper). `ext_adv_coef` is fixed to 1.
- `gamma_min, gamma_max: 0.99 and 0.997`, like the NGU paper.
- `entropy_coef = 0.01`: I found 0.01 performed better than 0.05 with the NGU PPO.
- `learn_step = 512, batchsize = 256, lambda = 0.95, epoch = 10, lr = 7e-5, target_kl = 0.05, clip_param = 0.2, max_grad_norm = 0.5, norm_adv = false, V_coef = 0.5`, as in previous projects.

#### Policies, gamma and beta

NGU uses 32 independent policies corresponding to 32 environments (instead of using a shared policy like other algorithms). Note: data for a specific policy (collected from its corresponding environment) is only used to train that policy (the NGU paper tried to train using shared data, but it didn't perform as well). Each policy in NGU receives additional gamma and beta parameters as input to the model to predict actions. Below are the gamma and beta graphs for policies 1-32 (similar to the NGU paper):

<div align="center">
  <table border="0">
    <tr align="center">
      <td>
        <img src="../figure/beta.png" alt="Beta Plot" width="400" height="250"/>
      </td>
      <td>
        <img src="../figure/gamma.png" alt="Gamma Plot" width="400" height="250"/>
      </td>
    </tr>
    <tr align="center">
      <td><b>Beta</b></td>
      <td><b>Gamma</b></td>
    </tr>
  </table>
</div>

As beta increases, policies prioritize exploration because the intrinsic advantage coefficient is higher. Intrinsic rewards become more important, so the model needs to explore more to increase the total intrinsic reward.

As gamma decreases, policies tend to explore to optimize short-term rewards, as subsequent steps contribute almost nothing due to the small multiplier for gamma.

From 0-31, the higher the env, the more policies prioritize exploration (because the beta coefficient is higher and gamma is lower than the preceding policies). I determined the gamma and beta for each policy as in the NGU paper (see image above).

Similar to NGU, I only used policy 0 to test the results. Note: different models will produce different behavioral sequences; you can experiment further yourself.

#### Reference time and step count table

Note: rerunning or using different seeds will yield different results

| World | Stage | training_step | training_time    |
|-------|-------|---------------|------------------|
| 1     | 1     | 218109        | 5:36:32          |
| 1     | 2     | 1208318       | 1 day, 18:12:30  |
| 1     | 3     | 3472384       | 2 days, 22:42:33 |
| 1     | 4     | 29161         | 0:44:58          |
| 2     | 1     | 552925        | 15:03:45         |
| 2     | 2     | 890875        | 19:33:53         |
| 2     | 3     | 150525        | 3:46:09          |
| 2     | 4     | 37887         | 0:58:11          |
| 3     | 1     | 244707        | 6:08:20          |
| 3     | 2     | 66035         | 1:45:30          |
| 3     | 3     | 182271        | 3:37:21          |
| 3     | 4     | 49150         | 1:15:11          |
| 4     | 1     | 96254         | 2:33:59          |
| 4     | 2     | 125950        | 3:52:43          |
| 4     | 3     | 49657         | 1:05:36          |
| 4     | 4     | 137718        | 3:31:47          |
| 5     | 1     | 195067        | 4:57:26          |
| 5     | 2     | 409075        | 9:11:11          |
| 5     | 3     | 345080        | 6:52:40          |
| 5     | 4     | 62970         | 1:37:49          |
| 6     | 1     | 57343         | 1:29:03          |
| 6     | 2     | 293374        | 9:56:34          |
| 6     | 3     | 215038        | 4:42:09          |
| 6     | 4     | 72700         | 1:38:13          |
| 7     | 1     | 295935        | 5:25:32          |
| 7     | 2     | 1410024       | 1 day, 11:52:23  |
| 7     | 3     | 159741        | 4:23:58          |
| 7     | 4     | 165886        | 3:40:42          |
| 8     | 1     | 648166        | 19:03:25         |
| 8     | 2     | 578045        | 15:34:57         |
| 8     | 3     | 700928        | 18:30:07         |
| 8     | 4     | 3538398       | 4 days, 21:44:42 |