## survey of relation extraction
関係抽出に関するサーベイのまとめ。
Temporal Relation Extraction (時間に関する関係抽出) は扱わない。

### Paper
#### Supervised Learning
- Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification, ACL, 2016 (short paper).
[[pdf]](http://www.aclweb.org/anthology/P/P16/P16-2034.pdf)
  - Attention機構を持つ双方向LSTMを始めてRelation Extractionに適用した論文
  - 前処理などを必要としないところが特長

- End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures, ACL, 2016.
[[pdf]](https://aclweb.org/anthology/P/P16/P16-1105.pdf)
  - 単語の語順および木構造を考慮可能なネットワークを提案
  - 木構造を考えることができるネットワークを始めてRelation Extractionに適用した論文
  - 係り受け解析は独立に行っている。→ End-to-Endでモデル化できないか。

- Relation Classification via Multi-Level Attention CNNs, ACL, 2016.
[[pdf]](http://aclweb.org/anthology/P/P16/P16-1123.pdf)
  - kk

- Semantic Relation Classification via Hierarchical Recurrent Neural Network with Attention, COLING, 2016.
[[pdf]](http://aclweb.org/anthology/C/C16/C16-1119.pdf)

- A Unified Architecture for Semantic Role Labeling and Relation Classification, COLING, 2016.
[[pdf]](http://aclweb.org/anthology/C/C16/C16-1120.pdf)
  - 意味役割付与と関係抽出を同時にニューラルネットワークでモデル化。
  - 係り受け解析と関係抽出を同時にモデル化できないか。

- Improved Relation Classification by Deep Recurrent Neural Networks with Data Augmentation, COLING, 2016.
[[pdf]](http://aclweb.org/anthology/C/C16/C16-1138.pdf)
  - 関係抽出におけるニューラルネットワークの適用事例は浅い構造が多い。
  - 始めて深い構造のネットワークで関係抽出した。
  - それに加えてData Augmentationも行っている。

- Pairwise Relation Classification with Mirror Instances and a Combined Convolutional Neural Network, COLING, 2016.
[[pdf]](http://aclweb.org/anthology/C/C16/C16-1223.pdf)

- Attention-Based Convolutional Neural Network for Semantic Relation Extraction, COLING, 2016.
[[pdf]](http://aclweb.org/anthology/C/C16/C16-1238.pdf)

- Table Filling Multi-Task Recurrent Neural Network for Joint Entity and Relation Extraction, COLING, 2016.
[[pdf]](http://aclweb.org/anthology/C/C16/C16-1239.pdf)



#### Distant supervision
オントロジーで関係がある2つのentityが出現する文は、その関係を述べている文として扱うことができるという仮説に基づき、学習データを自動で作成する。
Supervisedと違い、学習データ中にノイズが存在している(=wrong labeling problem)ので、それをどう扱うかが研究の焦点になることが多い。

- Neural Relation Extraction with Selective Attention over Instances, ACL, 2016.
[[pdf]](http://www.aclweb.org/anthology/P/P16/P16-1200.pdf)
  - Distant Supervisionのwrong labeling problemの解決を狙った論文
  - 他のニューラルモデルと比較すると、entityペアが含まれる全文を参照しているところが新しい。

- Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks, COLING, 2016.
[[pdf]](https://www.aclweb.org/anthology/C/C16/C16-1139.pdf)
  - Distant SupervisionにおけるMulti-instance Multi-labelingの問題の解決を狙った論文
  - この問題をCNNを用いて解いていることが新しい。

- Multi-instance Multi-label Learning for Relation Extraction
[[pdf]](http://ai2-s2-pdfs.s3.amazonaws.com/151e/e8aedc97e7a388a8edd704ff13698a7af0b4.pdf)

- Knowledge-based weak supervision for information extraction of overlapping relations, ACL, 2011.
[[pdf]](http://raphaelhoffmann.com/publications/acl2011.pdf)

- Modeling relations and their mentions without labeled text
[[pdf]](https://pdfs.semanticscholar.org/db55/0f7af299157c67d7f1874bf784dca10ce4a9.pdf)


#### Unsupervised
2つのentity間の動詞句を関係とする。
- Unsupervised Relation Discovery with Sense Disambiguation, ACL, 2012.
[[pdf]](http://www.aclweb.org/anthology/P12-1075)


#### Bootstrapping
種表現を基に、関係を表すパターンを自動で増やしていく。
思想はDistant Supervisionに似ている気がする。


#### Hand-built Patterns
人出でパターンをあらかじめ作成。
そのパターンを基に関係抽出。


### Dataset
- Supervised Relation Extraction
  - SemEval2010 Task 8 [[url]](http://www.kozareva.com/downloads.html)
  - ACE04, ACE05

- Distant Supervision
  - New York Times corpus (NYT)[[url]](http://iesl.cs.umass.edu/riedel/ecml/)
  - filtered version of NYT10 [[zip]](http://www.nlpr.ia.ac.cn/cip/ ̃liukang/liukangPageFile/code/ds_pcnns-master.zip)


### slide
- Makoto Miwa, Learning for Information Extraction in Biomedical and General Domains.
[[pdf]](http://www.toyota-ti.ac.jp/Lab/Denshi/COIN/people/makoto.miwa/docs/keynotetalk_biotxtm2016.pdf)

- Bill MacCartney, Relation Extraction.
[[pdf]](https://web.stanford.edu/class/cs224u/materials/cs224u-2016-relation-extraction.pdf)

- 海野さん, 情報抽出入門.
[[pdf]](http://www.slideshare.net/unnonouno/ss-21254386)