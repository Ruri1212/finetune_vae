## AutoEncoderのDecoderのみをチューニングするコード

このディレクトリは、AutoEncoder（特に、Stable DiffusionのようなDiffusion Modelで使われているもの）のDecoderをファインチューニングすることを目的に作られたディレクトリです。

ネットを検索しても、使いやすそうなものがなかったので自作しました。（超初心者が記述していものなので、中身の修正点がありましたら教えていただきたいです。）

> 参考にしたサイト：https://github.com/kukaiN/vae_finetune

実行環境としては、GPUメモリが24GBあれば、画像(1024*1024)でバッチサイズ1で学習ができました。
コードをいじればもっとメモリ容量を減らすことができると思います。

### 実行コマンド
```
(python 3.10以降)
pip install -r requirements.txt
accelerate launch --num_processes=1 train_vae.py --xformers --gradient_checkpointing
```

### ディレクトリ構造の説明
- `train_data/`：学習画像を保存するディレクトリ。（画像のパスをtrain_image_path.txtに記述するため階層構造は自由）

- `train_image_path.txt`：学習画像のパスを記述するテキストファイル。サンプルのように画像のパスを1行ずつ記述すれば良い。

- `train_decoder_vae.py`：decoderを学習するコード。（なるべく、改造がしやすいようにシンプルに書いたつもり。）