## survey of relation extraction
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

#### Distant supervision
- Neural Relation Extraction with Selective Attention over Instances, ACL, 2016.
[[pdf]](http://www.aclweb.org/anthology/P/P16/P16-1200.pdf)
  - Distant Supervisionのwrong labeling problemの解決を狙った論文
  - 他のニューラルモデルと比較すると、entityペアが含まれる全文を参照しているところが新しい。

- Relation Extraction with Multi-instance Multi-label Convolutional Neural Networks
[[pdf]](https://www.aclweb.org/anthology/C/C16/C16-1139.pdf)

- Multi-instance Multi-label Learning for Relation Extraction
[[pdf]](http://ai2-s2-pdfs.s3.amazonaws.com/151e/e8aedc97e7a388a8edd704ff13698a7af0b4.pdf)

- Knowledge-based weak supervision for information extraction of overlapping relations
[[pdf]](http://raphaelhoffmann.com/publications/acl2011.pdf)

- Modeling relations and their mentions without labeled text
[[pdf]](https://pdfs.semanticscholar.org/db55/0f7af299157c67d7f1874bf784dca10ce4a9.pdf)





### Dataset
- Supervised Relation Extraction
  - SemEval2010 Task 8 [[url]](http://www.kozareva.com/downloads.html)

- Distant Supervision
  - New York Times corpus (NYT)[[url]](http://iesl.cs.umass.edu/riedel/ecml/)
  - filtered version of NYT10 [[zip]](http://www.nlpr.ia.ac.cn/cip/ ̃liukang/liukangPageFile/code/ds_pcnns-master.zip)


### slide
- Makoto Miwa. Learning for Information Extraction in Biomedical and General Domains.[[pdf]](http://www.toyota-ti.ac.jp/Lab/Denshi/COIN/people/makoto.miwa/docs/keynotetalk_biotxtm2016.pdf)