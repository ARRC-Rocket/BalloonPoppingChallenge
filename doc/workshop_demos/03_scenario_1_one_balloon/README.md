# Demo 3：Scenario 1，只追一顆移動氣球

這個 demo 把目標刻意限制為「戳破一顆」。它保留 Scenario 1 最小但完整的 GNC
迴圈：觀察氣球、選擇可達目標、預測攔截點、估測姿態、用 TVC 追蹤。固定 seed
的預期結果是 **1 / 100**。

本 demo 的三個主要檔案：

- Agent：[`scenario_1_one_balloon_agent.py`](../../../BalloonPoppingGymEnv/agents/scenario_1_one_balloon_agent.py)
- Config：[`workshop_scenario_1_one_balloon.yaml`](../../../BalloonPoppingGymEnv/evaluation/configs/workshop_scenario_1_one_balloon.yaml)
- 教材：本 README

這不是把 6-balloon 競賽 Agent 換一個數字。原競賽 Agent 約 1,158 行，包含多目標
路線規劃、replanning 與機會目標；本版約 257 行，只保留單一目標需要的預測、
導引與控制，選定目標後不再換目標。

## 0. 物理：命中移動目標要瞄準「未來」

Scenario 1 有 100 顆氣球，每 0.5 秒釋放一顆。氣球的位置有隨機差異，並在真實
大氣資料的風中移動。火箭若只朝氣球「現在的位置」飛，到達時氣球通常已經離開。

在短時間內先用等加速度模型預測氣球：

```text
未來位置 = 現在位置 + 速度 × t + 1/2 × 加速度 × t²
```

若火箭從位置 `p`、速度 `v` 出發，希望 `t` 秒後到達預測位置 `p_target`，需要的
平均加速度可寫成：

```text
a_required = 2 × (p_target - p - v × t) / t²
```

但火箭推力還要對抗重力，所以真正希望推力指向：

```text
a_thrust = a_required - gravity
```

這仍是近似：真實火箭有阻力、質量變化、姿態動態與 actuator limits，氣球也不會
完美等加速度。因此 Agent 每 0.01 秒重新觀測並修正，而不是只在發射時算一次。

### 為什麼等待到 49.5 秒才發射？

等待讓更多氣球釋放，也累積連續速度樣本來估計氣球加速度。發射太早可選的目標
少；等太久則剩餘模擬時間與燃料裕度變少。`49.5` 是這個固定 seed 教學案例的簡單
可重現設定，不是所有 seed 的最佳答案。

## 1. Code 思路：把 G、N、C 分開看

### Guidance：選一顆，之後不換

`_choose_target()` 對每顆已釋放氣球嘗試 7、8、9、10、11 秒的 flight time，
排除超過推力、仰角太低或 closing speed 太高的組合，再偏好較短且仍有推力裕度
的攔截。選定後只保存：

- `target`：唯一目標 index。
- `intercept_time`：預計相遇時間。
- `launch_angles`：由推力方向換算出的 inclination 與 heading。

`_guidance()` 在飛行中持續重算需要的加速度；距離小於 35 m 時，再用相對位置與
相對速度估計 closest approach，補一個有限幅的 miss correction。

### Navigation：從 gyro 維護姿態

Agent 看不到 `info["rocket_states"]` 的真實姿態。它從發射角建立初始 quaternion，
再由 `integrate_attitude()` 積分 gyro，得到 body frame 到 launch frame 的旋轉。
這是最小的 dead reckoning；正式高難度 Agent 還會處理 bias、noise 與 sensor
fusion。

### Control：把期望加速度變成合法 action

`_control()` 先比較目前火箭軸與期望推力軸，再產生期望角速度；內層沿用 Demo 2
的想法，以 gyro rate error 產生 TVC，roll 則壓回 0。TVC/roll 同時限制最大值與
每 timestep 的變化量。

整體資料流是：

```text
balloon observation ──> target prediction ──> desired acceleration
gyro ──> attitude estimate ──────────────────> axis/rate control ──> TVC + roll
GNSS position/velocity ──────────────────────> guidance
```

## 2. 每段教學

### 2.1 只改 config，先看問題變了什麼（5 分鐘）

比較兩份 YAML：

```shell
git diff --no-index \
  BalloonPoppingGymEnv/evaluation/configs/workshop_ten_balloon.yaml \
  BalloonPoppingGymEnv/evaluation/configs/workshop_scenario_1_one_balloon.yaml
```

最重要的差異是 `scenario_number: 1`、Agent class 與延後的 launch time。先讓學員
預測：若直接把 Demo 2 的垂直穩定器放進 Scenario 1，為什麼很可能是 0 分？

### 2.2 讀 balloon observation（10 分鐘）

辨認：

- `balloon_status`：0 未釋放、1 可追、2 已戳破。
- `balloon_states[i, :3]`：第 i 顆的位置 `[x, y, z]`。
- `balloon_states[i, 3:6]`：第 i 顆的速度 `[vx, vy, vz]`。

用相鄰兩個 timestep 的速度差除以 `dt` 估計加速度，再用低通更新減少瞬間變化。

### 2.3 選一顆可達目標（15 分鐘）

逐行完成 `_predict_balloon()` 與 `_choose_target()` 的三個 feasibility checks。
請學員回答：只挑最近的氣球，為何可能需要過低的發射仰角或過大的 closing speed？

### 2.4 將攔截向量換成發射角（10 分鐘）

由單位推力向量 `axis` 計算：

```text
inclination = asin(axis_z)
heading = atan2(axis_x, axis_y)
```

這個專案的 X 是 east、Y 是 north，所以 heading 使用 `atan2(x, y)`；這和數學課
常見的 `atan2(y, x)` 不同，是很值得現場讓學員踩一次的座標系陷阱。

### 2.5 接上 navigation 與 control（20 分鐘）

先把 `_guidance()` 的輸出固定為垂直，確認控制器能工作；再接回預測的目標加速度。
用 `rotation_matrix()` 將軸誤差換到 body frame，最後重用 Demo 2 的 gyro rate
feedback。這段的學習目標是理解資料如何流動，不要求現場從記憶手寫 quaternion。

### 2.6 跑完整評估（15 分鐘）

```shell
uv run --no-sync python BalloonPoppingGymEnv/evaluation/evaluate.py \
  BalloonPoppingGymEnv/evaluation/configs/workshop_scenario_1_one_balloon.yaml
```

第一次執行 Scenario 1 會先建立 100 條氣球 Monte Carlo 軌跡，輸出較多且耗時較長。
結尾應看到：

```text
Total reward: 1
```

### 2.7 小實驗與 Takeaway（15 分鐘）

1. 將 `launch_time` 提前或延後 5 秒，觀察 target 與得分是否改變。
2. 只用氣球現在位置、不加入 `velocity × t`，驗證追尾誤差。
3. 移除 near-target correction，比較最接近距離與最後是否命中。

- Stabilization、guidance、navigation、control 是不同問題，可以分層驗證。
- 移動目標要預測相遇點；追現在的位置通常已經太晚。
- 一顆可重現的命中，就是可以量測、可以理解、可以再擴充的 Scenario 1 baseline。
- 下一步才是 target switching 與多目標 route planning，不需要在第一個 Agent 一次完成。
