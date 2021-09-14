# CIET.5<sub>embed</sub>
Contextual Information Extraction Technique based on 5 steps and using word embedding based models (CIET.5<sub>embed</sub>) is a new context extraction technique based on vector space model. Analogous to the [CIRT.5<sub>embed</sub>](https://github.com/joao8tunes/CIRT.5_embed), this technique assumes that the frequency relationship between terms is dependent, considering the reliance of a set of correlated terms (context) directly proportional to the frequency with his terms occurs in a text document. 

![](https://joao8tunes.github.io/hello/wp-content/uploads/photo-gallery/exemplo_etapa4.png?bwg=1542306867)

This CIET.5<sub>embed</sub> based script allows to convert a collection of text documents formed by terms into a collection of text documents formed by contexts. The main differences with other textual enrichment procedures such as named entity recognition and word sense disambiguation is that CIET.5<sub>embed</sub> based contexts extracted considers the local influence of textual scopes, in addition to enabling the volume and quality of information in texts through external knowledge sources like Wikipedia.

> Extracting a CIET.5<sub>embed</sub> based set of contexts:
```
python3 CIET.5_embed.py --language EN --contexts 3 --thresholds 0.05 --model models/model --input in/db/ --output out/CIET.5_embed/txt/
```


# Related scripts
* [CIET.5_embed.py](https://github.com/joao8tunes/CIET.5_embed/blob/master/CIET.5_embed.py)


# Assumptions
These script expect a database folder following an specific hierarchy like shown below:
```
in/db/                 (main directory)
---> class_1/          (class_1's directory)
---------> file_1      (text file)
---------> file_2      (text file)
---------> ...
---> class_2/          (class_2's directory)
---------> file_1      (text file)
---------> file_2      (text file)
---------> ...
---> ...
```


# Requirements installation (Linux)
> Python 3 + PIP installation as super user:
```
apt-get install python3 python3-pip
```
> Gensim installation as normal user:
```
pip3 install --upgrade gensim
```
> NLTK + Scipy + Numpy installation as normal user:
```
pip3 install -U nltk scipy numpy
```


# See more
Project page on LABIC website: http://sites.labic.icmc.usp.br/MSc-Thesis_Antunes_2018
