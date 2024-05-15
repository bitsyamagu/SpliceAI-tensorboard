SpliceAIを利用したいけど、最新のGPUではパフォーマンスが出ないという問題があって、ボトルネックを確認する必要が出てきた。まずは、プロファイルを取ろうということになり、PythonのcProfileやNVIDIAのNsightなどが利用可能だが、これまでにNsightなどによって計算時間の半分はTensorflow、残り半分はPython3のロジックで消費されていることまでは判明している。

SpliceAIはtensorflow上で動作するDeepLearningのソフトウェアで、ご存じの通りスプライシングジャンクション付近に入った変異のスプライシングへの影響を予測するソフトである。学習済みのモデルが公式のGitHubから公開されているので、それがそのまま利用できる。

Tensorflow専用のプロファイラとしては、Tensorboardがよく知られていて、主に学習用にということで機能が充実している。ところが今回の私のケースでは大量に予測を行いたいので、どれだけ予測の１GPUあたりの並列度を上げられるかや、予測の計算速度の向上が目的となっているので、少しTensorboardの用途とは異なるらしい。とはいえTensorflow上で動作するものであれば、Tensorflowの機能を利用して取得したプロファイルをTensorboardで評価できるようになっていた模様。

本来的には1台のホスト内にtensorflowのプログラムとtensorboardを同居させて、動作させながら状況を確認できるツールのようだが、今回のように予測のみの場合は、SpliceAIのようなtensorflowアプリケーションを動作させてプロファイリングを取得し、その終了後に別途Tensotboardで評価ということができる模様。

その方が記録も残しやすいし、最悪依存性の問題でSpliceAIのような解析プログラムとtensorboardが同居できなくても別のコンテナなどで評価できることになるので、良いかもしれない。
