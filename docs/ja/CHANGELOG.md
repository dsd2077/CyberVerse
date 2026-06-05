# 変更履歴

## 2026-05-29~2026-06-05

Features:
- AutoDL プラットフォーム用イメージを追加（審査中）
- LiveAct に FP4 GEMM 高速化を追加し、B シリーズ GPU で推論演算をさらに圧縮可能に（[2cace72](https://github.com/dsd2077/CyberVerse/commit/2cace72)）
- MuseTalk モデルの初期実装を導入。学習、前処理、オフライン/リアルタイム推論スクリプト、Gradio demo を含む（[c826645](https://github.com/dsd2077/CyberVerse/commit/c826645)）

Changes:
- PersonaAgent から `wait_for_more_input` ツールを削除。表現が不明瞭な場合はツール呼び出しではなく自然に追問（[aa691ee](https://github.com/dsd2077/CyberVerse/commit/aa691ee)）
- サンプル設定の `idle_strategy` デフォルトを `silent_inference` に変更し、連続推論モードと整合（[7644430](https://github.com/dsd2077/CyberVerse/commit/7644430)）

Bugs:
- LiveAct マルチ GPU 分散制御の broadcast が GPU 上で busy-wait する問題を修正。制御チャネルを独立した `gloo` プロセスグループに変更（[PR #19](https://github.com/dsd2077/CyberVerse/pull/19)、[54635a3](https://github.com/dsd2077/CyberVerse/commit/54635a3)）

Others:
- README に FAQ（QA）セクションを追加。RTP 指標でデジタルヒューマンのカクつきと音画遅延を診断可能に（[f7a1a4b](https://github.com/dsd2077/CyberVerse/commit/f7a1a4b)）
- LiveAct FP4 GEMM のビルド/インストール手順を追加（LightX2V / CUTLASS）（[ccbb978](https://github.com/dsd2077/CyberVerse/commit/ccbb978)）
- 音声処理のシステム依存を追加：`libopus-dev`、`libopusfile-dev`、`libsoxr-dev`、`pkg-config`（[dba745b](https://github.com/dsd2077/CyberVerse/commit/dba745b)）
- FlashAttention のインストール手順を 2.8.1 に更新（[74d58e6](https://github.com/dsd2077/CyberVerse/commit/74d58e6)）

## 2026-05-22~2026-05-29

今週は主に、リアルタイムデジタルヒューマンのパイプラインにおける Direct WebRTC の安定性、AV 同期、アイドル状態のデジタルヒューマン戦略、音声コンテキスト復元に取り組みました。

### fix: 音声と映像の同期ずれを修正

以前のバージョンには、会話が続くにつれて音声と映像が徐々に同期しなくなる問題がありました。この問題を修正するため、今週は `WebRTC` プロトコルをかなり詳しく調べ、原因を整理しました。原因は複雑ですが、大きくは次の3つです。

1. 送信側の pacing が不安定だった
2. turn 間の idle gap の扱いが誤っていた
3. ブラウザ receiver の状態が turn をまたいで残っていた

この3つの問題を理解するには、まず CyberVerse がどのように音声と映像をフロントエンドへ送っているかを見る必要があります。`cached_video` モードでは、デジタルヒューマンは turn ごとに話します。デジタルヒューマンが話していないときは、事前に生成された動画を再生します。そのため、このモードは `cached_video` と呼ばれています。動画ストリーム全体は、おおよそ次のようになります。

```text
speaking -> idle -> speaking -> idle -> speaking
```

間の停止時間は固定ではありません。数秒の場合もあれば、数十秒になることもあります。問題はここにありました。私は同じ Direct WebRTC media path を使って、分割されたリアルタイム音声・映像を運んでいましたが、この media path の送信リズム、RTP タイムライン、ブラウザ receiver 状態は、以前は「連続したライブストリーム」のように扱われていました。

1つ目の原因は、送信側の pacing が不安定だったことです。以前、サーバーが動画を publish するときは、「映像フレームを1枚書き込み、その後 frameDur だけ sleep する」という相対的なリズムを使っていました。この実装は単純に見えますが、パケット書き込み、スケジューリング、sleep のたびに誤差が発生し、その誤差は session 内でフレームごとに蓄積します。さらに厄介だったのは、Opus 音声フレームが 20 ms のリズムで滑らかに送信されず、映像フレームごとにまとめて送信されていたことです。実際には、音声も映像フレームに合わせてまとまって送られていました。ブラウザが受け取る arrival pattern は不均一になり、動画 jitter buffer が徐々に大きくなり、最終的には映像が音声よりどんどん遅れる症状として現れました。

2つ目の原因は、speaking turn 間の RTP idle gap の扱いが正しくなかったことです。2つの speaking turn の間の停止は 2 秒の場合もあれば、2 分の場合もあります。しかし以前の RTP timestamp gap correction は、最大でも 2 秒分しかスキップしていませんでした。ブラウザから見ると、これは非常に異常な信号になります。実際のネットワーク到着時間では長い時間が経過しているのに、RTP media clock は 2 秒しか進んでいないからです。ブラウザの video jitter buffer は後続のパケットを深刻に遅延したストリームとして扱い、`video_jb` が汚染されると、その後の再生にも影響し続けました。

3つ目の原因は、ブラウザ receiver の状態が speaking turn をまたいで残ることです。フロントエンドが WebRTC の映像から idle 動画へ切り替えても、それは見た目のレイヤーでの切り替えにすぎません。`RTCPeerConnection`、receiver、jitter buffer はリセットされません。つまり、ある speaking turn で動画 jitter buffer が高くなってしまうと、その後の turn では生成側がすでに正常に戻っていても、この異常な状態を引き継ぐ可能性があります。これが、ログ上である turn から明らかな drift が始まり、その後の turn でも `JBDelta(window)` が高いまま残っていた理由です。

最初は vibe coding でこの問題を解こうとしましたが、丸一日試しても解決できませんでした。音声と映像の同期ずれは「主観的」な判断であり、AI はそれを直接知覚できません。そのため、AI がどれだけ論理的に推論しても、「同期していない」という状態そのものは知覚できません。そこで、パイプライン全体に監視を追加し、音声と映像の同期差を定量化する方法を調査しました。[Commit ad470a7](https://github.com/dsd2077/CyberVerse/commit/ad470a7652ce39345941f304aaa091fe35519695)。実際のフィードバック指標が得られると、AI には調整の方向性ができました。

原因が明確になった後は、修正自体は単純でした。[Commit 0ad5730](https://github.com/dsd2077/CyberVerse/commit/0ad5730641d36e2de1ed575d71c3ca3cb510c997)

### feat: `silent_inference` 推論モードを追加

CyberVerse はこれまで常に `cached_video` モードを採用していました。このモードにはいくつか利点があります。計算リソースを節約でき、デジタルヒューマンの演出余地が広がり、ユーザーはより強力な動画生成モデルを使って、見栄えの良い待機動画を多数生成できます。一方で、このモードには欠点もあります。人物全体の連続性が十分ではなく、「話していない -> 話す -> 話していない」という過程で画面に断絶が出ます。そこで、この問題を解決するために `silent_inference` モードを追加しました。このモードではモデルが常に推論を続け、話しているときは音声 audio で推論し、話していないときは silent audio で推論します。「話していない -> 話す -> 話していない」という過程が非常に連続的になり、フレーム飛びは発生しません。もちろん欠点もあります。話していないときの画面はずっと繰り返され、その画面を制御することはできません。silent driving でも random noise driving でも、大きな違いはありません。[Commit 57512f7](https://github.com/dsd2077/CyberVerse/commit/57512f77b21f07ff2a8b7b3af184159eaa9883bc)

### fix: 音声パイプラインとコンテキスト復元

タイムアウト後の再接続でコンテキストが失われる問題を修正しました。Qwen Omni モデルには 5 分の接続タイムアウトが設定されています。タイムアウト後は、再度ファイルを送信して wake up できます。しかし再接続後に、前の turn のコンテキストが引き継がれない bug がありました。[Commit 0e1c171](https://github.com/dsd2077/CyberVerse/commit/0e1c171c41fc28543f199df7629a14ac5b1078de)

### エンジニアリング規約ドキュメントの調整

- AGENTS.md / Claude.md に Truth-First Reasoning Rules を追加し、AI が自分自身で議論し続けるような状況を抑えるようにしました。詳細: [X](https://x.com/dsd2077/status/2059101010317189519?s=20)
- andrej-karpathy-skills の4原則を削除しました。ネット上のフィードバックでは、このプロンプトはあまり有用ではなく、すでに公式の挙動に取り込まれている可能性があるとのことでした。

[Commit e6788a8](https://github.com/dsd2077/CyberVerse/commit/e6788a843ed806d78a3268fe9fee1854855a8639)
