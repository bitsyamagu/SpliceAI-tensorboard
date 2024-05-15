SpliceAIを利用したいけど、最新のGPUではパフォーマンスが出ないという問題があって、ボトルネックを確認する必要が出てきた。まずは、プロファイルを取ろうということになり、PythonのcProfileやNVIDIAのNsightなどが利用可能だが、これまでにNsightなどによって計算時間の半分はTensorflow、残り半分はPython3のロジックで消費されていることまでは判明している。

SpliceAIはtensorflow上で動作するDeepLearningのソフトウェアで、ご存じの通りスプライシングジャンクション付近に入った変異のスプライシングへの影響を予測するソフトである。学習済みのモデルが公式のGitHubから公開されているので、それがそのまま利用できる。

Tensorflow専用のプロファイラとしては、Tensorboardがよく知られていて、主に学習用にということで機能が充実している。ところが今回の私のケースでは大量に予測を行いたいので、どれだけ予測の１GPUあたりの並列度を上げられるかや、予測の計算速度の向上が目的となっているので、少しTensorboardの用途とは異なるらしい。とはいえTensorflow上で動作するものであれば、Tensorflowの機能を利用して取得したプロファイルをTensorboardで評価できるようになっていた模様。

本来的には1台のホスト内にtensorflowのプログラムとtensorboardを同居させて、動作させながら状況を確認できるツールのようだが、今回のように予測のみの場合は、SpliceAIのようなtensorflowアプリケーションを動作させてプロファイリングを取得し、その終了後に別途Tensotboardで評価ということができる模様。

その方が記録も残しやすいし、最悪依存性の問題でSpliceAIのような解析プログラムとtensorboardが同居できなくても別のコンテナなどで評価できることになるので、良いかもしれない。
NVIDIA関係のパッケージはバージョンの互換性の条件がCPU版よりも厳しい印象なので、できるだけ依存性は
切り離して最小限に抑えておかないと構築しきれない。

# 環境構築手順
## 1. Dockerイメージの作成
## 1.1 プロファイリングの開始・終了の埋め込み
予測(model.predict())のプロファイリングを行うには、まず予測を行っている箇所を
プロファイラの開始と終了で囲う。SpliceAIの場合はutils.py内部にこのような箇所があるので
右のように修正する。

![image](https://github.com/bitsyamagu/SpliceAI-tensorboard/assets/49506866/cfae2726-bc2a-48ca-bd65-98c75a68391e)

開始時に指定している"logs"はログの格納先ディレクトリで、実行時にはこの下に階層的に
ディレクトリが作られて各種ログが保存される。

### 1.2 以下のようなDockerfileを用意する。
Dockerfile:
```
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    git \
    libtinfo5 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

RUN /opt/conda/bin/conda create -n spliceai python=3.8 && \
    /opt/conda/bin/conda install -n spliceai -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

RUN /opt/conda/bin/conda run -n spliceai pip install --upgrade pip setuptools wheel
RUN /opt/conda/bin/conda run -n spliceai pip install numpy==1.21.6 absl-py==1.0.0 flatbuffers==2.0 grpcio==1.34.0 tensorflow-estimator==2.11.0 pyfaidx==0.5.9.5 pysam==0.16.0.1 tensorflow-gpu==2.11.0 tensorboard==2.11.0 tensorboard_plugin_profile protobuf==3.19.6


RUN git clone https://github.com/Illumina/SpliceAI.git /SpliceAI
COPY utils.py /SpliceAI/spliceai/utils.py
RUN cd /SpliceAI && \
    /opt/conda/bin/conda run -n spliceai pip install .

WORKDIR /SpliceAI
```
このDockerfileはコンテナ内部にMinicondaを利用してtensorboardの環境とtensorflowアプリケーション(SpliceAI)をインストールするだけのものである。
SpliceAIにつてはGitHubから取得後、先のプロファイリング用に改変したutils.pyでオリジナルを上書きしてからインストールしている。

## 1.3 コンテナイメージをビルドする
Dockerfileのあるディレクトリで以下のようにビルドする。
```
docker build -t spliceai-tensorboard .
```
# 2. Tensorboard
## 2.1 Tensorboardの起動スクリプトを用意しておく
run_tensorboard.sh:
```
#!/bin/bash

# TensorBoardの起動
/opt/conda/bin/conda run -n spliceai tensorboard --logdir=/logs --host=0.0.0.0
```
## 2.2 Tensorboardを起動する
以下のようなコマンドで起動できる
```
docker run --gpus all -it -p 80:6006 -v /disk3/spliceai:/disk3/spliceai \
    -v /ssd3/parabricks:/ssd3/parabricks \
    -v /home/yamagu/logs:/logs \
    spliceai-tensorboard /bin/bash /disk3/spliceai/run_tensorboard.sh
```
- /home/yamagu/logsはホスト側のログ出力先
- /disk3/spliceaiは、tensorboard起動スクリプトを指定したディレクトリを指定する
- parabricksのフォルダは本件とは直接関係無いが、後でSpliceAIがコンテナにattachしてリファレンスゲノムのファイルをそこから読み込むことができるようにしておくため。
- -p 80:6006は、Dockerコンテナ内のTensorboradがlistenする6006番ポートを、ホスト外からも手軽にWebで見られるように80番ポートにマップして公開

# 3. SpliceAI(model.predict())を動かして測定
## 3.1 Tensorboardのコンテナにattach
```
docker ps
```
既にコンテナが止まっている場合は```docker ps -a```で確認する。ここで表示されたコンテナIDを控えておく。
```

コンテナが既に止まっている場合はstartしてからexecで。
```
docker start [コンテナID] 
```
コンテナに入ってSpliceAIを実行
```
docker exec -it [コンテナID] /bin/bash
/opt/conda/envs/spliceai/bin/spliceai -I /disk3/spliceai/chr20_2000.vcf \
  -O /disk3/spliceai/chr20_2000_snv_spliceai.vcf \
  -R /ssd3/parabricks/Homo_sapiens.GRCh38.dna_sm.primary_assembly_fix.fa \
  -A /disk3/spliceai/gencode45_grch38.txt
```

# Tensorboardにアクセス
ブラウザで、htttp://(ホスト名またはIPアドレス)/ にアクセスする
