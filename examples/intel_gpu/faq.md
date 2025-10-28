## check x11 or wayland

```
echo $XDG_SESSION_TYPE
```

## check intel gpu for pytorch

```
python -c 'import torch; print(torch.xpu.is_available())'
```

## leader will be ttyACM2 and follower will be ttyACM1

**❗please insert follow usb and then leader usb**

```
sudo chmod 777 /dev/ttyACM*
```

## calibration leader

```
lerobot-calibrate   \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM2 \
  --teleop.id=ssp_leader_arm_01
```

## calibration follower

```
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=ssp_follower_arm_01

```

## teleoperation

```
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=ssp_follower_arm_01 \
    --robot.cameras="{ top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, griper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader  \
    --teleop.port=/dev/ttyACM2  \
    --teleop.id=ssp_leader_arm_01 \
    --display_data=true

```

## record

```
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=ssp_follower_arm_01 \
    --robot.cameras="{ top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, griper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader  \
    --teleop.port=/dev/ttyACM2  \
    --teleop.id=ssp_leader_arm_01 \
    --display_data=true \
    --dataset.repo_id=hzsunxuan/record-test \
    --dataset.num_episodes=20 \
    --dataset.push_to_hub=False \
    --dataset.single_task="put block in box"
```

## train act

```
lerobot-train \
    --dataset.repo_id=hzsunxuan/record-act-1014  \
    --policy.type=act   --output_dir=outputs/train/act_so101_act_1014 \
    --job_name=act_so101_1014   --policy.device=cuda \
    --policy.repo_id=hzsunxuan/act_policy1014
```

## eval act policy and inference

```
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=ssp_follower_arm_01 \
    --robot.cameras="{ top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, griper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --display_data=false \
    --dataset.repo_id=hzsunxuan/eval_act_so101_test \
    --dataset.push_to_hub=False \
    --display_data=true \
    --dataset.num_episodes=10 \
    --dataset.single_task="put block in box" \
    --policy.device=xpu \
    --policy.path=/home/robot/sunausti/train/act_so101_act_1014_b60/checkpoints/last/pretrained_model/
```

```
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=ssp_follower_arm_01 \
    --robot.cameras="{ top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, griper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --display_data=false \
    --dataset.repo_id=hzsunxuan/eval_act_so101_test \
    --dataset.push_to_hub=False \
    --display_data=true \
    --dataset.num_episodes=10 \
    --dataset.single_task="put the scotch tape in box" \
    --policy.device=xpu \
    --policy.path=/local/sunausti/train_result/pi05_training_1016/checkpoints/last/pretrained_model/
```

## train pi05 on 4090D

```
lerobot-train     --dataset.repo_id=hzsunxuan/record-act-1016     --policy.type=pi05     --output_dir=./outputs/pi05_training_nocompile_float32     --job_name=pi05_training     --policy.repo_id=hzsunxuan/pi05_policy1015     --policy.pretrained_path=lerobot/pi05_base     --policy.compile_model=false     --policy.gradient_checkpointing=true     --wandb.enable=false     --policy.dtype=bfloat16     --steps=3000     --policy.device=cuda     --batch_size=16 --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
```

## policy run pi05 
```
lerobot-record     --robot.type=so101_follower     --robot.port=/dev/ttyACM1     --robot.id=ssp_follower_arm_01     --robot.cameras="{ top: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, griper: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"     --display_data=false     --dataset.repo_id=hzsunxuan/eval_pi05_so101_$(date +"%m_%d_%H_%M")  --dataset.push_to_hub=False     --dataset.num_episodes=10     --dataset.single_task="put the scotch tape in box"     --policy.device=xpu     --policy.path=/local/sunausti/train_result/pi05_training_1016/checkpoints/last/pretrained_model/ --policy.compile_model=false --policy.dtype=bfloat16
```

