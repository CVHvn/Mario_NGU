# MARIO_NGU
Playing Super Mario Bros using NEVER GIVE UP (NGU)

## Introduction

My PyTorch NEVER GIVE UP ([NGU](https://arxiv.org/pdf/2002.06038)) implement to playing Super Mario Bros. My implementation use NGU as intrinsic reward and combine it with [PPO](https://arxiv.org/abs/1707.06347) instead of [R2D2](https://openreview.net/pdf?id=r1lyTjAqYX) as original NGU (R2D2 base).

Intrinsic reward of NGU include two terms:
- Term 1: Episodic memory and intrinsic reward (some paper ([RLeXplore](https://arxiv.org/pdf/2405.19548)) call this "pseudo-counts")
- Term 2: Integrating life-long curiosity (just normalize RND reward):
    - Almost RND implement (include original paper) just devide RND output to running std of RND reward or return of RND reward.
    - But NGU subtract running mean and devide by running std.
- I test three versions of NGU intrinsic reward: 
    - 1 policy only pseudo-counts (without RND). Please note that, all method attempts to estimate the number of times a state occurs are called "pseudo-counts". "Pseudo-counts" is not a proprietary name for this algorithm. When referring to "pseudo-counts" in this project, it will by default refer to NGU pseudo-counts.
    - 1 policy NGU (with both pseudo-counts and RND)
    - many policy with NGU (UVFA framework, each actor use different gamma, beta as NGU paper, than each actor have 1 unique policy, only 1 model but with different input (gamma, beta) yeild different policy for each actor) 
- Because:
    - Original paper test three versions!
    - RLeXplore show that full NGU (with RND) maybe poor performance than pseudo-counts (NGU without RND). Please note that we can't sure  RLeXplore show correct analysis because:
        - I don't know RLeXplore use correct hyperparameters and correct implementations because NGU don't have public code!
        - They just test in some envs.
        - They just finetune some hyperparameter set (maybe suboptimal).

## Motivation

Original NGU implement base on R2D2 but I can't find True NGU or even R2D2 available opensource. Almost project just simpler version of NGU or R2D2. They don't use different epsilon values in epsilon greedy, different gamma and different beta for each actor like NGU paper. Some project maybe implement incorrect or suboptimal make performance of NGU very poor when compare with other intrinsic reward method can easy implement (easy than no bug!) or have reputable opensource.

I read this paper and want to reimplement this to find stronger intrinsic reward can make PPO learn better.

I use Super Mario Bros to test NGU because I can compare NGU with many other algorithms I implemented before. And I still find algorithm that can solve all stages of SMB without finetune hyperparameters. With 1 set of hyperparameters, I will train all stages of SMB with 1 policy (if we need finetune hyperparameters to complete some stages, we can't complete all stages with 1 policy!).

## Requirements

* **python 3>3.6**
* **gym==0.25.2**
* **gym-super-mario-bros==7.4.0**
* **imageio**
* **imageio-ffmpeg**
* **cv2**
* **pytorch** 
* **numpy**

## Acknowledgements
With my code, I can completed all 32/32 stages of Super Mario Bros. 

## Reference

* [NGU paper](https://arxiv.org/pdf/2002.06038)
* [PPO paper](https://arxiv.org/abs/1707.06347)
* [Coac NGU](https://github.com/Coac/never-give-up/tree/main)
* [CVHvn PPO](https://github.com/CVHvn/Mario_PPO)
* [Stable-baseline3 PPO](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO)
* [lazyprogrammer A2C](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl3/a2c)
* [jcwleo RND](https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/utils.py)
* [DI-engine RND](https://opendilab.github.io/DI-engine/12_policies/rnd.html)
* [vwxyzjn cleanrl/ppo_rnd_envpool.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py)