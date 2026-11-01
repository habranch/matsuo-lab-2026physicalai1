# Physical AI応用1講座 最終課題

## ロボットアームの障害物回避と、実演データの集め方の比較

MuJoCo内のCRANE+ V2ロボットアームに、壁を越えて目標へ到達する動作を、模倣学習モデルACTで学習させる実験コードです。同じ20回の実演を一つの開始姿勢から集める場合と、5種類の開始姿勢へ分ける場合を比較します。

実演は関節角度の指令を段階的に変えるプログラムで生成します。モデルの入力は4つの関節角度、出力は関節角度の指令です。画像や壁・目標の座標は入力しません。

## 実験条件

| 条件 | 実演の開始角 | 実演数 |
| --- | --- | --- |
| B：同一姿勢 | 0° | 20回 |
| C：複数姿勢 | −10°、−5°、0°、+5°、+10° | 各4回、計20回 |
| A：少量データ | 0°。Bの最初の5回を使用 | 5回 |

開始角は、根元を水平方向へ回す関節1の、基準姿勢からの角度差です。Aは実演数を減らした比較条件で、主な比較対象はBとCです。各条件で学習シード0〜4の5モデルを学習します。

- 制御周期は20Hz。ACTは20時刻、1秒分の指令を予測し、時間的アンサンブルで実行します。
- 学習のバッチサイズは64、学習率は0.0001。基本評価は2,000回更新したモデル、追加評価は20,000回更新したモデルを使用します。基本評価の更新回数は、本評価とは別の8状態を使った予備評価で設定した値です。
- 基本評価は共通の25状態で実施します。実演と同じ開始姿勢1状態と、開始姿勢として未使用の24状態を含み、各条件125試行、A・B・C合計375試行です。
- 25秒以内に先端を目標から2cm以内へ到達させ、その範囲を連続0.5秒間維持すると成功です。壁への接触は失敗、25秒で時間切れとなります。
- 壁と目標の位置は固定です。壁接触、机接触、自己貫通を監視します。自己衝突による反力は計算しません。
- 比較の基準として、実演1回分の指令をそのまま再生する方法も評価します。開始姿勢の範囲を広げた14状態の評価は、2,000回更新したモデルを使用する探索的な追加評価です。

## 構成

| パス | 内容 |
| --- | --- |
| `src/crane_reach/` | シミュレータ、実演生成、データ変換、開始姿勢、推論・評価ループ |
| `scripts/` | データ準備、学習、評価、結果集計、一括実行 |
| `configs/task.yaml` | 制御周期、目標、開始姿勢、成功条件 |
| `configs/train_act.yaml` | ACTの学習・推論設定 |
| `worlds/` | MuJoCoモデル、ロボットのメッシュ、検証用URDF |
| `exp/000_smoke/run_checks.py` | 運動学、物理動作、接触判定、データ時刻の検証 |
| `exp/002_obstacle/pilot_states.csv` | 予備評価の8状態 |
| `exp/003_demo_diversity/evaluation_states.csv` | 基本評価の25状態 |
| `exp/004_wide_range/states.csv` | 探索的な追加評価の14状態 |

実演データ、学習済みモデル、軌跡は実行時に`artifacts/`へ保存されます。試行ごとの結果CSVと集計図は`exp/<実験名>/results/`に生成されます。これらの生成物はGitの管理対象外です。

## 実行環境

Linux、Python 3.10、CUDA対応GPUを使います。MuJoCo 3.3.0、PyTorch 2.6.0と、固定コミットのLeRobotを使用し、依存関係は`uv.lock`に記録しています。

```bash
uv sync --locked
```

## 実験全体の実行

実行順序は、モデルとデータ形式の検証、開始状態の作成、実演生成、A・B・C各5モデルの学習、基本評価・追加評価・指令再生、集計です。ジョブは指定したGPU上で順番に実行されます。

```bash
# 実行するコマンドの一覧
python3 scripts/reproduce.py --gpu 0 --dry-run

# 全工程
uv run python scripts/reproduce.py --gpu 0
```

工程を指定した実行例：

```bash
uv run python scripts/reproduce.py --gpu 0 --stages prepare
uv run python scripts/reproduce.py --gpu 0 --stages train
uv run python scripts/reproduce.py --gpu 0 --stages evaluate analyze
```

`--skip-existing`は、必要な出力がすべて存在するジョブを省略します。コード・設定・入力データが同じ場合の再実行用です。既存の学習フォルダを上書きする処理や、中断した学習の自動再開はありません。別条件での学習は異なる`--run-id`で管理します。

## 1モデルを個別に実行する例

```bash
# 実演データとA・B・Cのデータセットを生成
uv run python scripts/collect_demos.py

# Bのシード0を学習
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_act.py \
  --dataset B --seed 0 --run-id 003_B_seed0

# 2,000回更新したモデルを25状態で評価
CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate.py \
  --run-id 003_B_seed0 --model step_002000 \
  --states exp/003_demo_diversity/evaluation_states.csv \
  --eval-id 003_B_seed0_eval

# 学習済みモデルを使わず、実演指令を再生
uv run python scripts/evaluate.py --replay B000 --device cpu \
  --states exp/003_demo_diversity/evaluation_states.csv \
  --eval-id 003_replayB000_eval

# 実行済みの評価を集計
uv run python scripts/analyze.py
```

## 出力と評価指標

`collect_demos.py`は実演ごとの条件と採否を`data_manifest.csv`、データ量を`dataset_summary.csv`に記録します。学習フォルダには設定・シード・依存バージョン・データのハッシュ・チェックポイントを保存します。

`evaluate.py`は成功、壁接触、時間切れ、試行終了時の目標までの距離、成功までの時間、机接触、自己貫通を記録します。終了時の距離を測る時点は終了理由によって異なります。各条件125試行は5つの学習モデルの評価であり、125個の独立した学習モデルの比較ではありません。

`analyze.py`は開始角・学習シード・終了理由ごとの集計CSVと図を生成します。基本評価と20,000回更新の追加評価は別々に集計されます。

## モデルと参照コード

CRANE+ V2のモデル素材とURDFは[rt-net/crane_plus](https://github.com/rt-net/crane_plus)を使用しています。ACTは[Hugging Face LeRobot](https://github.com/huggingface/lerobot)の固定コミットを使用します。学習・データ形式と関節操作の参照元は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)に記載しています。
