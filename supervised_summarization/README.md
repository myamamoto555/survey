# Neural Networkベースの手法 
## Abstractive Summarization
### 概要

### データセット


### 参考文献
- "Abstractive Summarization with Attentive Recurrent Neural Networks", NAACL-HLT2016.
[[pdf]](http://nlp.seas.harvard.edu/papers/naacl16_summary.pdf)
- "Incorporating Copying Mechanism in Sequence-to-Sequence Learning", ACL2016.
[[pdf]](https://www.aclweb.org/anthology/P/P16/P16-1154.pdf)
- "Pointing the Unknown Words", ACL2016.
[[pdf]](https://www.aclweb.org/anthology/P/P16/P16-1014.pdf)
- "A Neural Attention Model for Abstractive Sentence Summarization", EMNLP2015.
[[pdf]](https://www.aclweb.org/anthology/D/D15/D15-1044.pdf)
- "Neural Headline Generation with Minimum Risk Training", Arxiv2016.
[[url]](https://128.84.21.199/abs/1604.01904v1)
- "Sequence-to-Sequence RNNs for Text Summarization", Arxiv2016.
[[url]](https://128.84.21.199/abs/1602.06023v1)
- "Sequence Level Training with Recurrent Neural Networks", Arxiv2015.
[[url]](https://arxiv.org/abs/1511.06732)
- "Generating News Headlines with Recurrent Neural Networks", Arxiv2015.
[[url]](https://arxiv.org/abs/1512.01712)

# Neural Network以外の手法
## 分類問題として解く
### 概要
大きく分けて2つのステップに分かれる。第一に特徴量の抽出、第二に分類器の構築である。

### 特徴量の抽出
様々な特徴量の抽出の仕方が考えられる。
- 文の長さ
- 文のスコア
  - 文書とのcosine類似度などで定義
- 文の位置
- 手がかり語のあるなし
  - "In summary" や "as a conclusion"が文中に入っているかなど
- 誰が話しているか
- 談話関係の特徴量
- 音声特徴量
- 自動音声認識結果の確信度
- コンテクスト特徴量
  - 近い文の特徴量を使用

### 分類器の構築
分類器についても、多様な選択肢が考えられる。代表的なものは以下の通りである。
- SVM
- Bayesian Network
- Maximum Entropy
- CRF
- 線形回帰
- 多層パーセプトロン

### 参考文献
- G.Tur et al., "Spoken Language Understanding: Systems for Extracting Semantic Information from Speech", Wiley, 2011. (会社でお借りしていた本。この本を中心にまとめました。より詳しく知りたい場合は、この本を読むのがいいと思います。)