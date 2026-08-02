# Demo 1：垂直發射，先戳破 7 顆氣球

這個 demo 的目標不是一次解完控制問題，而是先跑通「讀取 observation、回傳
action、由 evaluator 計分」的完整流程。固定 seed 的預期結果是 **7 / 10**。

本 demo 的三個檔案：

- Agent：[`workshop_agent.py`](../../../BalloonPoppingGymEnv/agents/workshop_agent.py)
- Config：[`workshop_vertical_baseline.yaml`](../../../BalloonPoppingGymEnv/evaluation/configs/workshop_vertical_baseline.yaml)
- 教材：本 README

## 0. 物理：為什麼火箭會向上飛？

火箭把燃氣高速向下噴出，燃氣也對火箭施加大小相等、方向相反的力。當向上的
推力大於向下的重力與阻力，火箭就會向上加速：

```text
合力 = 推力 - 重力 - 阻力
加速度 = 合力 / 質量
```

Scenario 0 把 10 顆半徑 1.5 m 的靜止氣球排在發射點正上方：第一顆在場地上方
10 m，之後每 40 m 一顆。因此最直覺的策略就是把發射仰角設成 90°、全油門
垂直飛行。

不過「一開始朝上」不等於「永遠直直向上」。微小旋轉會讓推力產生水平分量，
誤差會隨飛行時間累積，所以這個完全不看感測器的開迴路策略只命中前 7 顆。
這正好建立下一個 demo 要改善的 baseline。

## 1. Code 思路：先做最小可用 Agent

Evaluator 每個 timestep 呼叫一次 `get_action(observation)`。這版只讀模擬時間，
然後固定輸出五個命令：

```python
return {
    "launch": simulation_time >= self.launch_time,
    "launch_inclination_heading": np.array([90.0, 0.0]),
    "tvc": np.array([0.0, 0.0]),
    "roll": 0.0,
    "throttle": self.throttle,
}
```

- `launch`：1 秒後發射。
- `launch_inclination_heading`：仰角 90°；垂直時 heading 不影響方向。
- `tvc`：引擎不偏轉，推力沿火箭本體軸線。
- `roll`：不施加滾轉力矩。
- `throttle`：全油門；初始化時仍用 scenario 公開的範圍做 clipping。

這是 open-loop control：輸出只依時間決定，火箭偏掉也不會修正。

## 2. 每段教學

### 2.1 找到入口與設定檔（5 分鐘）

先開啟 config，辨認 `scenario_number`、Agent 檔案、class 名稱與 constructor
參數。提醒學員：YAML 負責「選哪個情境、載入哪個 Agent」，控制邏輯仍在
Python。

### 2.2 逐欄認識 observation 與 action（10 分鐘）

從 `BaseAgent` 的介面開始，請學員找出這版唯一使用的 observation 欄位
`simulation_time`。再對照上面的五個 action，特別確認仰角是從水平面量起，
因此 90° 才是向上。

### 2.3 跑 baseline（10 分鐘）

在 repository root 執行：

```shell
uv run --no-sync python BalloonPoppingGymEnv/evaluation/evaluate.py \
  BalloonPoppingGymEnv/evaluation/configs/workshop_vertical_baseline.yaml
```

結尾應看到：

```text
Total reward: 7
```

`render_mode: "matplotlib"` 會顯示軌跡；若只想快速計分，可暫時改成 `null`。

### 2.4 小實驗與討論（10 分鐘）

依序只改一個值再重跑：

1. 把 `launch_time` 改成 2 秒：Scenario 0 的靜止氣球是否受影響？
2. 把 `throttle` 改成 0.8：最高高度與得分如何改變？
3. 把仰角改成 88°：很小的初始角度誤差為何會越飛越遠？

### 2.5 Takeaway（5 分鐘）

- 先建立可量測的 baseline，才能知道後續控制是否真的改善結果。
- 開迴路很容易理解，但不會修正擾動與累積誤差。
- 下一個 demo 只加入必要的 gyro feedback，目標從 7 / 10 提升到 10 / 10。
