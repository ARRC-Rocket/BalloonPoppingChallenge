# RocketPy CG／CP 與 Weathercocking Demo

這份 demo 使用比賽 scenario YAML 的火箭、HybridMotor、氧化劑槽、鼻錐與尾翼參數，讓使用者只調整少數變數，就能比較 CG、CP、static margin 與有風飛行軌跡。

Notebook 刻意不加入 S1 agent 的閉迴路導引：馬達維持 YAML 的 nominal full thrust，TVC 與 roll torque 固定為零。這讓不同 static margin 的被動 weathercocking 差異不會被控制器掩蓋。

## 檔案

- [`rocketpy_competition_demo.ipynb`](./rocketpy_competition_demo.ipynb)：使用者操作、圖表與輸出。
- [`rocketpy_demo_setup.py`](./rocketpy_demo_setup.py)：讀取 YAML 並建立 RocketPy 物件；一般使用者不需修改。
- `rocketpy_demo_outputs/`：執行後產生的 PNG、CSV、KML 與 JSON；此資料夾不納入 Git。

## 啟動

在 repository root 執行：

```shell
uv sync --extra notebook
uv run jupyter lab doc/workshop_demos/00_rocketpy_competition_demo/rocketpy_competition_demo.ipynb
```

第一次開啟 notebook 請執行一次 **Run All**。之後只要修改「1. 使用者控制區」，再從該格執行 **Run All Below**。

## 最常使用的設定

```python
# 最推薦：直接指定點火時 static margin
TARGET_INITIAL_STATIC_MARGIN_CAL = 2.0
TARGET_INITIAL_CG_CP_GAP_M = None

# None 使用執行當天台灣時間 02:00；指定 date 可固定比較風場
FORECAST_DATE_TW = None
FORECAST_TIME_TW = None

# East × North × AGL 的完整視野；發射點位於水平面的中心
TRAJECTORY_VIEW_SIZE_M = (250.0, 250.0, 200.0)
```

預設水平範圍是 East `-125～+125 m`、North `-125～+125 m`，各自完整跨距為 250 m；軌跡圖會直接顯示這兩個端點。

`TARGET_INITIAL_STATIC_MARGIN_CAL` 與 `TARGET_INITIAL_CG_CP_GAP_M` 只能擇一。正的 `CG - CP` 和正的 static margin 代表靜態穩定。

Static-margin target 的實作會移動本體 CG，但保留原本 CP 與 inertia。極端數值適合做敏感度分析，不代表已完成可製造的配重與慣量設計。

## 與比賽的關係

對齊的項目：

- scenario YAML 火箭與推進參數
- S0 baseline 的 90° 發射姿態
- 0.01 m 虛擬導軌
- RK45、scenario time step
- S0／S1 的 100／150 秒模擬上限

刻意保留的差異：

- S0 使用無風 standard atmosphere。
- S1 agent 會依目標選擇發射角，並在固定 Ensemble 風場中逐步控制 TVC。
- 本 demo 預設使用執行當天 02:00 的 GFS，且不加入閉迴路控制器。

若要重現 S0 的無風垂直參考，只需設定：

```python
WEATHER_MODEL = "standard_atmosphere"
LAUNCH_INCLINATION_DEG = 90.0
```

同一火箭在此無風設定下實測可飛約 58 秒、達約 1.59 km AGL，因此有風 neutral-controls 案例提前落地不是燃料不足。相同推力下增加推進劑會降低推重比。

## 閱讀輸出

模擬前會先列印 CG、CP、`CG - CP` 和 static margin，並顯示 RocketPy 的火箭配置圖。每次 Run All 會產生：

- `00_rocket_layout_cg_cp`：火箭、CG 與 CP
- `01_wind_profile`：風速、East／North 分量與氣象風向
- `02_stability`：燃燒期間 CG、CP 與 static margin
- `03_trajectory`：固定視野的 3D 軌跡與 ground track
- `04_attitude_angles`：flight-path、attitude 與 lateral-attitude angle
- `05_performance`：高度、速度、Mach、dynamic pressure 與 angle of attack
- `competition_rocket_trajectory.kml`：Google Earth 軌跡
- `competition_rocket_flight.csv`：飛行資料
- `run_metadata.json`：本次設定與摘要

所有檔案存放在同一個 `rocketpy_demo_outputs/`，並使用實際 margin 與台灣時間戳，例如：

```text
03_trajectory_2.00cal_20260806_143000.png
```

氣象風向表示「風從哪裡來」。例如 `from 281°` 是風從西北偏西吹來、往約 101° 的東南偏東流動；weathercocking 則是火箭鼻端朝來風方向偏轉。

## 建議比較流程

1. 固定 GFS 日期與時間。
2. 依序測試 `0.5`、`1.0`、`2.0`、`3.0` cal。
3. 比較相同尾綴群組中的 wind、stability、trajectory、attitude 與 performance 圖。
4. 再切換 `standard_atmosphere`，分離風場與 static margin 的影響。

這是設計與教學工具，不是飛行安全認證。實體火箭仍需驗證結構、致動器、風場不確定性、感測器、控制器、製造公差與 Monte Carlo 結果。
