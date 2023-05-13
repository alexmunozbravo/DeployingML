---
annotations_creators:
- found
language:
- en
language_creators:
- found
license:
- apache-2.0
multilinguality:
- monolingual
pretty_name: Veiled toxicity dataset
size_categories:
- 10K<n<100K
source_datasets:
- original
tags: []
task_categories:
- text-classification
task_ids:
- sentiment-analysis
---

# Dataset Card for Veiled Toxicity dataset

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)

## Dataset Description

- **Homepage:** https://github.com/xhan77/veiled-toxicity-detection
- **Repository:** https://github.com/xhan77/veiled-toxicity-detection
- **Paper:** https://arxiv.org/abs/2010.03154
- **Leaderboard:** None
- **Point of Contact:** xiaochuh@cs.cmu.edu, ytsvetko@cs.cmu.edu

### Dataset Summary

This dataset was created by Han & Tsvetkov in their paper "Fortifying Toxic Speech detectors against veiled toxicity"
to train and evaluate a model for detecting veiled toxic speech. The dataset consists of 15000 comments from various sources
annotated automatically with PerspectiveAPI as well as three human annotators to determine its toxicity level.

### Supported Tasks and Leaderboards

Supported Tasks: Sentiment Analysis, Text-classification, Multi-label classification

### Languages

Monolingual - English

## Dataset Structure

### Data Instances

#### plain_text
- **Size of downloaded dataset files:** 65.5 MB
- **Size of the generated dataset:** 3.11 MB
- **Total amount of disk used:** 68.61 MB
An example of 'train' pickle files looks as follows.
```
{
    "text_a": "This rice eaters suck\n",
    "community": ["asian folks", "Chinese people"]
}
```

### Data Fields

The data fields are the same among all sub-datasets and splits.
#### plain_text
- `text_a`: a `string` feature.
- `text_b`: a `string` feature, typically None.
- `label`: a classification label, with possible values including `pos`(0), `neg` (1).
- `note`: a `string` feature.
- `guid`: an `integer` feature.

### Data Splits

The core dataset contains 15,000 reviews split into 12k train
and 3k test sets. The overall distribution of classes is as follows:
- 2000 examples for veiled toxicity in training and 1000 in test.
- 8000 examples for positive class in training and 1000 in test.
- 2000 examples for overt toxicity in training and 1000 in test.

## Dataset Creation

### Curation Rationale

This dataset is a byproduct of Han & Tsvetkov (2020) work, and it was
designed to provide an elegant approach to design more robust datasets
for a niche task such as veiled toxicity detection, where the collection
of data can be expensive both in resources and time. The dataset was originally designed for sentiment analysis binary classification but a "community" attribute is also present in it, opening the possibility for
multi-label classification.

### Source Data

#### Initial Data Collection and Normalization

This dataset was created by following this procedure:
1.	They use two sources for the construction of the dataset:
a.	SBIC dataset: Contains 45k social media posts with crowdsourced annotations of offensiveness, intention, and targeted group from a variety of origins. Every post has three attributes: 
i.	Binary offensiveness label (hateful or not) 
ii.	Offensive score (Between 0 and 1, it is the mean of the scores given by the annotators)
iii.	Target communities (it is free text and can be more than one community).
b.	Randomly sampling 10K general reddits from no specific domains.
2.	From the SBIC dataset, select those posts with offensive scores> 0.5 (this means that more than half of the annotators thought it was offensive). From those, keep only the ones that have at least one target community.
3.	Measure the general toxicity from Reddit posts with PerspectiveAPI (Lees, 2022), a toxic speech detector produced by Google and Jigsaw that is widely used as a baseline for hate speech detection. This tool will be used as the labeler of our observations, and it is expected to produce incorrect labels for the veiled toxicity examples. 
The output of PerspectiveAPI is composed of the label and a number comprised between 0 and 1 that tells how toxic is the example. Measuring the mean toxicity value of the Reddit posts we get a value of general toxicity equal to 0.17, this value will be used in the following steps to process the remaining dataset.
4.	Measure the toxicity values for every post selected from the SBIC dataset and select the least m toxic ones such that the mean of their toxicity values is equal to the one obtained from the Reddit posts. This yields around 3k results, 2k being used for training and 1k for testing.
5.	Extract all SBIC posts that are annotated as not offensive and get their toxicity scores. Select the least m’ posts such that their toxicity is again equal to the general toxicity value obtained from Reddit. This yields a total of around 10k posts that are considered non-hateful/toxic, 8k are selected for training, and 1k for testing.
6.	Select the posts identified as offensive and extract those with a toxicity score > 0.8 (this threshold is recommended by PerspectiveAPI). This yields around 3k posts, from which 2k are selected for training and 1k for testing. This is considered as the overtly-hateful set, or in other words, examples that most of the standard classifiers could determine as toxic “easily”.

#### Who are the source language producers?

The language producers are users of various online forums and social media
such as Twitter or Reddit.

### Annotations

The dataset contains communities affected by online comments but no extra annotations on the thought process behind it.

#### Annotation process

Every commment is annotated by three different annotators, the mean of their scores is the offensiveness value that the comment receives. The communities are conformed into a list that contains all of them (if that is the case).

#### Who are the annotators?

Crowdsourced annotations.

## Considerations for Using the Data

### Social Impact of Dataset

The purpose of this dataset is to help develop better toxic speech detectors, particularly to develop systems that can detect veiled toxicity in online scenarios.

A system that succeeds at the supported task would be able to provide a state of the art system that serves as the basis for online toxic speech detection, and more particularly, for the safety and well-being of minorities and dangered communities in online spaces.

### Discussion of Biases

It is widely known that all NLP tasks/systems include certain biases (both explicit and implicit) transmitted through language. In this case, since we are dealing with veiled toxicity annotated via crowdsourcing, it could be the case that a difficulty for labelling certain examples could exist. This could in turn be the case of mislabelling of certain comments, deviating the distribution present in the dataset from the real one. Moreover, since the dataset is quite limited in terms of scale, it could be that the subset seen by the models is not enough to detect other types of veiled toxicity that could exist.

## Additional Information

### Dataset Curators

This dataset was created by Xiaochuang Han and Yulia Tsvetkov during their work done for their paper Fortifying Toxic Speech Detectors Against Veiled toxicity (Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)) paper.

### Licensing Information

Although not explicitely stated, it is advised to cite the original paper when using this dataset. 