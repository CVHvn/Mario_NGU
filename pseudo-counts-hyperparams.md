# Pseudo-counts hyperparameters

## How to find hyperparameters

RL is sensitive with hyperparameters and I have limited resource to test hyperparameters. To find hyperparameters, I list hyperparameters used in NGU paper and combine with PPO-RND hyperparameters I did before:
- Intrinsic coef: 0.3 (or 0 --> 0.3 for multiple policy) --> because R2D2 != PPO, I can't use it --> I use 0.5 as my old RND and DRND projects.
- Last 5/80 frames use to train RND and embedding model --> update_proportion = 6.25% --> I round to update_proportion = 0.1
- The parameters for calculating episodic reward are the same as the paper: k=10, kernel_cluster_distance=0.008, kernel_epsilon=0.0001, c=0.001, sm=8 (see my compute_reward_int_pseudo_counts function)

I found that NGU gives better intrinsic reward and makes PPO_NGU less sensitive to hyperparameters than the old projects Please note that I only consider it ineffective after one or two unsuccessful runs. For an accurate assessment and comparison with other algorithms, multiple experiments and probabilities comparisons are needed (but I lack the resources and time for this project). So I fixed almost hyperparameters and only changed entropy_coef for special stages when the default parameters do not work well.

After having the default parameters, I played with the stages and adjusted the entropy_coef to 0.05 for the stages that couldn't be completed with the default entropy_coef = 0.01. For stage 8-4, I set num_env = 32.

## Default hyperparameter

Here is the default hyperparameter table, This table applies to all stages; the hyperparameters that change for specific stages will be listed later.

| Hyperparameters | Value |
| :--- | :--- |
| **num_envs** | 16 |
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
| **loss_type** | huber |
| **k** | 10 |
| **kernel_cluster_distance** | 0.008 |
| **kernel_epsilon** | 0.0001|
| **c** | 0.001 |
| **sm** | 8 |

Below is a table of hyperparameters for each specific stage

| World | Stage | num_envs   | entropy_coef | training_step | training_time   |
|-------|-------|------------|--------------|---------------|-----------------|
| default |     | 16         | 0.01         |               |                 |
| 1     | 1     | 16         | 0.01         | 181240        | 1:52:03         |
| 1     | 2     | 16         | 0.01         | 169440        | 2:02:06         |
| 1     | 3     | 16         | `0.05`       | 1702398       | 19:58:44        |
| 1     | 4     | 16         | 0.01         | 33749         | 0:22:52         |
| 2     | 1     | 16         | 0.01         | 778741        | 8:57:32         |
| 2     | 2     | 16         | 0.01         | 268779        | 2:53:32         |
| 2     | 3     | 16         | 0.01         | 110065        | 1:10:37         |
| 2     | 4     | 16         | 0.01         | 84988         | 0:50:12         |
| 3     | 1     | 16         | 0.01         | 503745        | 4:45:02         |
| 3     | 2     | 16         | 0.01         | 76792         | 0:52:16         |
| 3     | 3     | 16         | 0.01         | 53751         | 0:35:54         |
| 3     | 4     | 16         | 0.01         | 44538         | 0:27:24         |
| 4     | 1     | 16         | 0.01         | 101372        | 1:10:13         |
| 4     | 2     | 16         | 0.01         | 231909        | 2:49:56         |
| 4     | 3     | 16         | 0.01         | 130048        | 0:53:01         |
| 4     | 4     | 16         | 0.01         | 261632        | 2:50:15         |
| 5     | 1     | 16         | 0.01         | 209407        | 2:02:43         |
| 5     | 2     | 16         | 0.01         | 570365        | 6:47:22         |
| 5     | 3     | 16         | 0.01         | 3721706       | 1 day, 2:25:25  |
| 5     | 4     | 16         | 0.01         | 100803        | 1:09:53         |
| 6     | 1     | 16         | 0.01         | 72698         | 0:48:05         |
| 6     | 2     | 16         | 0.01         | 400376        | 4:17:12         |
| 6     | 3     | 16         | 0.01         | 4179968       | 1 day, 6:11:35  |
| 6     | 4     | 16         | 0.01         | 49151         | 0:32:24         |
| 7     | 1     | 16         | 0.01         | 551421        | 5:15:23         |
| 7     | 2     | 16         | 0.01         | 1771470       | 17:32:13        |
| 7     | 3     | 16         | 0.01         | 275448        | 2:48:15         |
| 7     | 4     | 16         | 0.01         | 285163        | 2:59:08         |
| 8     | 1     | 16         | 0.01         | 4681203       | 1 day, 11:42:07 |
| 8     | 2     | 16         | 0.01         | 1546237       | 15:32:07        |
| 8     | 3     | 16         | 0.01         | 863737        | 8:33:36         |
| 8     | 4     | `32`       | `0.05`       | 5104952       | 3 days, 4:39:03 |