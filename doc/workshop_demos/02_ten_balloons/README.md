# Demo 2：加入 gyro feedback，戳破 10 顆氣球

這個 demo 延續 Demo 1 的垂直發射；不改目標、不改油門，只加入角速度回授。
固定 seed 的預期結果是 **10 / 10**。

本 demo 的三個檔案：

- Agent：[`ten_balloon_agent.py`](../../../BalloonPoppingGymEnv/agents/ten_balloon_agent.py)
- Config：[`workshop_ten_balloon.yaml`](../../../BalloonPoppingGymEnv/evaluation/configs/workshop_ten_balloon.yaml)
- 教材：本 README

先比較兩個 Agent，會比從頭閱讀更容易看懂：

```shell
git diff --no-index \
  BalloonPoppingGymEnv/agents/workshop_agent.py \
  BalloonPoppingGymEnv/agents/ten_balloon_agent.py
```

## 0. 物理：控制的其實是推力方向

Demo 1 的火箭只要出現角速度，姿態就會持續改變；火箭一傾斜，原本向上的推力
便分成垂直與水平分量：

```text
垂直推力 = T cos(傾斜角)
水平推力 = T sin(傾斜角)
```

即使傾斜角很小，水平加速度經過兩次時間累積，仍會變成明顯的位置偏差。要飛過
同一條垂直氣球線，我們先採取一個簡單目標：讓三軸角速度都回到 0。

- pitch / yaw：用 TVC 改變推力方向，產生回正力矩。
- roll：TVC 不足以直接處理本體軸滾轉，因此使用獨立的 roll torque。
- gyro：量到的是 body frame 的角速度，單位為 rad/s。

這裡不是完整的姿態控制。角速度為 0 只代表「停止繼續轉」，不保證火箭已回到
絕對垂直；但對 Scenario 0 已足以展示回授如何抑制偏移。

## 1. Code 思路：在 Demo 1 上只加一個 feedback loop

Agent 先沿用 Demo 1 的發射與油門命令，再讀 `rocket_sensors[:3]` 的 gyro：

```text
rate_error = 0 - gyro
TVC = Kp × pitch/yaw rate_error
roll = Kp × roll rate_error + Ki × 累積誤差
```

程式另外處理三個真實控制器一定會遇到的細節：

1. 發射前 sensor 是 `NaN`，所以只在 gyro 有效時啟動回授。
2. 指令用 `np.clip` 限制最大 TVC、roll torque 與每個 timestep 的變化量。
3. roll 指令飽和時停止累積 integral，避免 integral windup。

Demo 1 的 action keys 與垂直發射設定保持不變；新增的程式只負責覆寫 `tvc` 與
`roll`。這是本系列刻意維持小 diff 的地方。

## 2. 每段教學

### 2.1 先看 diff，不重讀整份程式（5 分鐘）

執行上方 `git diff --no-index`，請學員標出三類新增內容：control limits、controller
state、gyro feedback。確認發射方向與油門仍和 Demo 1 相同。

### 2.2 讀 gyro 並處理發射前 NaN（10 分鐘）

請學員印出或檢查 `rocket_sensors[:3]`，觀察發射前後差異。討論為何直接把 NaN
送進 controller，最後會讓整個 action 都變成 NaN。

### 2.3 先完成 P controller（15 分鐘）

把目標角速度設成 0，完成 `rate_error = -gyro`，再將 pitch/yaw 誤差轉成 TVC。
提醒學員正負號應由實際系統反應驗證，並觀察角度與 rate clipping 何時生效。

### 2.4 為 roll 加入小型 PI（10 分鐘）

加入累積誤差，說明 P 項處理「現在偏多快」、I 項處理「長期是否仍有偏差」。
用 conditional integration 說明 actuator 已到極限時，繼續累加只會讓恢復更慢。

### 2.5 跑完整評估（10 分鐘）

```shell
uv run --no-sync python BalloonPoppingGymEnv/evaluation/evaluate.py \
  BalloonPoppingGymEnv/evaluation/configs/workshop_ten_balloon.yaml
```

結尾應看到：

```text
Total reward: 10
```

Config 預設 `render_mode: null`，避免視覺化拖慢反覆調參；想展示軌跡時再改成
`"matplotlib"`。

### 2.6 小實驗與 Takeaway（10 分鐘）

1. 將 `pitch_yaw_kp` 改小或改大，比較修正不足與指令飽和。
2. 暫時移除 `np.clip`，討論模擬器為何仍會限制 actuator，以及 Agent 自己限制
   指令的好處。
3. 暫時把 gyro feedback 關掉，確認分數回到 baseline 附近。

- Feedback 的核心循環是「量測 → 比較 → 修正」。
- 飽和、NaN 與積分 windup 不是附加題，而是實際控制程式的一部分。
- Scenario 0 只需穩定；Scenario 1 還要決定何時發射、追哪顆移動氣球。
