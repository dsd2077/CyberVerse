# CyberVerse

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語** | [한국어](README.ko.md)

### たった一枚の写真から、息づくデジタルヒューマンへ。

> あなたを本当に見て、聞いて、リアルタイムで話しかけてくれる、自分だけの J.A.R.V.I.S. のような AI を夢見たことはありませんか？
> もう会えない大切な人に再び会い、その声を聞き、笑顔を見ることができたらどうでしょうか。
> あるいは、ずっと命を吹き込みたかったキャラクターがいるかもしれません。
>
> **必要なのはたった一枚の写真。CyberVerse がその存在を動き出させます。**

CyberVerse は、リアルタイムのビデオ通話に対応したオープンソースの**デジタルヒューマン・エージェント・プラットフォーム**です。ビデオ通話のように、顔を見ながら会話できる AI エージェントを作成できます。

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

## 特長

### リアルタイムのビデオ通話

録画済みではありません。ターン制でもありません。デジタルヒューマンと**時間制限なく**低遅延でリアルタイムにビデオ通話でき、初回フレームは**約 1.5 秒**で表示されます。WebRTC を基盤とし、P2P ストリーミングと組み込みの TURN / NAT トラバーサルを備えています。

### アバターではなく、エージェント

各デジタルヒューマンは、ただ話せるアバターではありません。実際に物事をこなす AI です。

### 一枚の写真から生成

一枚の写真をアップロードするだけで、デジタルヒューマンを作成できます。業界最先端のデジタルヒューマンモデルにより、リアルタイムの表情アニメーション、自然なリップシンク、待機時の呼吸感を実現します。3D モデリングもモーションキャプチャも不要です。

### エージェントを組み立てる

頭脳、顔、声、耳。そのすべてのコンポーネントを差し替え可能なプラグインとして扱えます。YAML 設定で LLM、TTS エンジン、ASR モデル、アバターバックエンドを自由に組み合わせられます。GPT-4o、Doubao、OpenAI TTS、Whisper、FlashHead を標準でサポートしています。

## デモ

<div align="center">

| [![](docs/demo/爱丽丝.mov.png)](https://youtu.be/Lk88sew2x4o) | [![](docs/demo/丽娜.mov.png)](https://youtu.be/8jdQ3ThcwgA) |
|:---:|:---:|
| [**Alice — YouTube で見る**](https://youtu.be/Lk88sew2x4o) | [**Lina — YouTube で見る**](https://youtu.be/8jdQ3ThcwgA) |

| [![](docs/demo/小龙女.mov.png)](https://youtu.be/WjEHUYZx5Gs) |
|:---:|
| [**Xiaolongnü — YouTube で見る**](https://youtu.be/WjEHUYZx5Gs) |

</div>

## ハードウェア要件

リアルタイムのビデオ会話には GPU アクセラレーションが必要です。以下は FlashHead と LiveAct アバターモデルのベンチマークです。

| モデル | 品質 | GPU | 枚数 | 解像度 | FPS | リアルタイム可？ |
|-------|---------|-----|-------|------------|-----|------------|
| FlashHead 1.3B | Pro | RTX 5090 | 2 | 512×512 | 25+ | ✅ はい |
| FlashHead 1.3B | Pro | RTX 4090 | 1 | 512×512 | ~10.8 | ❌ いいえ |
| FlashHead 1.3B | Pro | RTX PRO 6000 | 1 | 512×512 | 20 | ✅ はい |
| FlashHead 1.3B | Lite | RTX 4090 | 1 | 512×512 | 25+ | ✅ はい |
| LiveAct 18B | — | RTX PRO 6000 | 2 | 320×480 | 20 | ✅ はい |
| LiveAct 18B | — | RTX PRO 6000 | 1 | 256×417 | 20 | ✅ はい |

> **Pro** は画質優先、**Lite** は速度優先です。表は代表的な **画質と計算資源のバランス** の例です。余裕があれば画質を上げられ、不足なら解像度や **Pro** / **Lite** など画質側の設定を下げてリアルタイム性を確保してください。

## クイックスタート

### 前提条件

- Python 3.10+
- Node 18+
- Go 1.22+
- PyTorch 2.8（CUDA 12.8）
- CUDA 12.8+ に対応した GPU
- FFmpeg（動画エンコードのため `libvpx` を含むこと）

### ステップ 1: クローンする

```bash
git clone https://github.com/anthropics/CyberVerse.git
cd CyberVerse
```

### ステップ 2: Python 環境を作成する

```bash
conda create -n cyberverse python=3.10
conda activate cyberverse
```

### ステップ 3: 環境変数を設定する

```bash
cp infra/.env.example .env
```

`.env` を編集して API キーを入力します。

```
DOUBAO_ACCESS_TOKEN=your_doubao_access_token   # ByteDance Doubao 音声 LLM
DOUBAO_APP_ID=your_doubao_app_id
```

Doubao Voice: [Volcengine クイックスタート](https://www.volcengine.com/docs/6561/2119699?lang=zh)に従い **App ID** / **API Key** を取得し、`DOUBAO_APP_ID` / `DOUBAO_ACCESS_TOKEN` に設定します。

スタック起動後は、これらの値や他の API キー / サービスエンドポイントも、`.env` を直接編集する代わりに Web UI の **`/settings`** から変更できます。

### ステップ 4: モデル重みをダウンロードする

CyberVerse は現在 **FlashHead** と **LiveAct** の 2 モデルに対応しています。必要なものだけダウンロードしてください。今後、さらにバックエンドを追加する予定です。

```bash
pip install "huggingface_hub[cli]"
```

#### FlashHead（SoulX-FlashHead）

| モデルコンポーネント | 説明 | リンク |
| :--- | :--- | :--- |
| `SoulX-FlashHead-1_3B` | 1.3B FlashHead 重み | [Hugging Face](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) |
| `wav2vec2-base-960h` | 音声特徴抽出器 | [Hugging Face](https://huggingface.co/facebook/wav2vec2-base-960h), [ModelScope](https://modelscope.cn/models/facebook/wav2vec2-base-960h) |

```bash
# 中国本土から利用する場合は、先にミラーを設定できます:
# export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
  --local-dir ./checkpoints/SoulX-FlashHead-1_3B

huggingface-cli download facebook/wav2vec2-base-960h \
  --local-dir ./checkpoints/wav2vec2-base-960h
```

#### LiveAct（SoulX-LiveAct）

| モデル名 | ダウンロード |
|-----------|----------|
| SoulX-LiveAct | [Hugging Face](https://huggingface.co/Soul-AILab/LiveAct), [ModelScope](https://modelscope.cn/models/Soul-AILab/LiveAct) |
| chinese-wav2vec2-base | [Hugging Face](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base), [ModelScope](https://modelscope.cn/models/TencentGameMate/chinese-wav2vec2-base) |

```bash
huggingface-cli download Soul-AILab/LiveAct \
  --local-dir ./checkpoints/LiveAct

huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
  --local-dir ./checkpoints/chinese-wav2vec2-base
```


### ステップ 5: 設定を更新する

`cyberverse_config.yaml` を編集し、モデルパスをローカルの checkpoint パスに合わせて更新します。

```yaml
inference:
  avatar:
    default: "flash_head"               # 起動するアバターモデルを指定。live_act を選ぶ場合は下の live_act 設定を記入
    runtime:
      cuda_visible_devices: 0      # 共有 GPU ID。マルチ GPU の場合は 0,1 など
      world_size: 1                # 共有 GPU 数。デュアル GPU なら 2
    flash_head:
      checkpoint_dir: "./checkpoints/SoulX-FlashHead-1_3B"  # ← ローカルのパス
      wav2vec_dir: "./checkpoints/wav2vec2-base-960h"        # ← ローカルのパス
      model_type: "lite"           # 高画質が必要なら "pro"（より多くの GPU が必要）
```

ここでのパス編集はひとまずスキップして、あとで Web UI から調整しても構いません。

### ステップ 6: SageAttention と FlashAttention をインストールする（任意）

```bash
# SageAttention
pip install sageattention==2.2.0 --no-build-isolation
```

```bash
# FlashAttention（任意）
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation
```

> コンパイルに時間がかかる場合は、[flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2) からビルド済み wheel をダウンロードし、`pip install <wheel>.whl` を実行してください。



### ステップ 7: プロジェクト依存関係をインストールする

```bash
make setup
```

これにより、基本の editable package（`[dev,inference]`）のインストール、gRPC stubs の生成、フロントエンド依存関係のインストールが行われます。追加の Python パッケージが必要な場合は、まとめて**全部**を入れるか（サイズ大）、[`pyproject.toml`](pyproject.toml) の `[project.optional-dependencies]` にある extras を必要なものだけ選んでください。

```bash
# すべての optional グループを一括でインストール
pip install -e ".[all]"

# または必要なものだけを選択。例:
pip install -e ".[voice_llm,flash_head]"
pip install -e ".[live_act]"
```

### ステップ 8: サービスを起動する（3 つのターミナル）

**ターミナル 1** — Python 推論サーバー:

```bash
conda activate cyberverse
make inference
```

`make inference` は `cyberverse_config.yaml` の `inference.avatar.default` を読み取り、現在の推論プロセスではその 1 つのアバターモデルだけを初期化します。起動ログには有効なアバターモデル名が表示されます。

次のログが出るまで待ちます。

- `Active avatar model initialized: <model_name>`
- `CyberVerse Inference Server started on port 50051`

**ターミナル 2** — Go API サーバー:

```bash
make server
```

**ターミナル 3** — フロントエンド:

```bash
make frontend
```

### ステップ 9: 確認する

```bash
# API ヘルスを確認
curl -s http://localhost:8080/api/v1/health
```

### リモートアクセス時は 8443/TCP の疎通を確認する

`streaming_mode: direct` で組み込み TURN を使う場合、ブラウザはサーバーの `8443/TCP` に到達できる必要があります。ページは開けるのに音声・映像がいつまでも接続されない、またはサーバーログに `ICE connection state: failed` や `publish timeout waiting for connection` が出る場合は、まず手元の端末からサーバーの `8443` ポートに疎通できるか確認してください。

```bash
nc -vz <server-ip> 8443
```

`8443` に到達できない場合、原因はクラウドのセキュリティグループ、ファイアウォール、または NAT 制限であることが一般的です。その場合は、SSH トンネルでローカルの `8443` をサーバーへ転送できます。

```bash
ssh -L 8443:127.0.0.1:8443 user@host -p port
```

トンネル確立後、ブラウザはローカルの `127.0.0.1:8443` 経由でリモート TURN サービスへ接続します。

SSH トンネルではなくブラウザからリモートサーバーへ直接接続したい場合は、`cyberverse_config.yaml` の `pipeline.ice_public_ip` にサーバーのグローバル IP またはドメインを設定してください。SSH トンネルを使う場合は、デフォルト値（`127.0.0.1`）のままで構いません。

ブラウザで http://localhost:5173 を開けば準備完了です。

## ロードマップ

### **デジタルヒューマン作成プラットフォーム**
キャラクターや推論設定を構成し、リアルタイムのデジタルヒューマンセッションを起動します。

- [x] 複数の参照画像、アクティブ画像、固定 / ランダム表示モード、任意の顔切り抜き、タグ、音声フィールド、人格、ウェルカムメッセージ、システムプロンプトを備えたキャラクター CRUD
- [x] 参照画像から、設定可能な Avatar プラグイン（FlashHead、LiveAct など）でリアルタイムのアバター映像を駆動
- [x] WebRTC によるリアルタイム音声・映像。直接 P2P（組み込み TURN）または LiveKit SFU
- [x] avatar、voice LLM、LLM、TTS、ASR をプラグインとして提供し、YAML で各ベンダーの API キーを設定可能（現状は豆包音声のキー 1 つで実行可能）
- [x] セッション管理：キャラクター単位で会話履歴をディスクに永続化し、会話開始時に読み込み
- [x] 音声クローン：豆包音声の音声クローンに対応
- [ ] モデル発話中の音声割り込みとセッションの中断・再開
- [ ] 知識・文書・人物の生平などの素材を取り込み、キャラクターに沿った RAG による回答
- [ ] Face-to-face：ユーザー側のカメラ / 映像入力と、動作・ジェスチャーなど視覚的手がかりの理解
- [ ] 開発者向けのサイト埋め込み（Web コンポーネントまたは SDK）、自己ホストしたインスタンスを自サイトへ接続
- [ ] ライブ配信向けの音声・映像ストリーミング

### 2. **デジタルヒューマン・エージェント**
デジタルヒューマンを、記憶・ツール・タスク実行能力を備えたエージェントへ進化させます。

- [ ] **記憶システム**：セッションをまたぐ長期記憶と、キャラクター知識ベース・RAG の連携により、人物背景と対話の一貫性を強化
- [ ] ツール利用と function calling を追加
- [ ] ワークフロー実行とタスク完了機能を追加

### 3. **エージェントネットワーク**
複数のエージェントを接続し、相互にコミュニケーションし、協調し、ネットワークを形成できるようにします。

- [ ] agent-to-agent 通信を有効化
- [ ] マルチエージェント協調と委譲を有効化
- [ ] エージェント間の共有メモリと共有知識を有効化
- [ ] 接続されたエージェントのオープンネットワークを構築

## ライセンス

GNU General Public License v3.0。詳細は [LICENSE](LICENSE) を参照してください。

## 謝辞

- [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) — Soul AI Lab によるアバターモデル

- [SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct) - Soul AI Lab によるアバターモデル
- [Pion](https://github.com/pion/webrtc) — Go の WebRTC 実装
