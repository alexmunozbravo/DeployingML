# Model Card for Recognizing veiled toxicity with deep learning.

## Persons and Organization
Alex Muñoz Bravo, Christian Hardmeier

## Contact

Please, you are welcome to adress questions and comments about the model to alex.munoz.bravo@estudiantat.upc.edu

## Table of Contents

- [1. Model Date and Version](#model-date-and-version)
- [2. Model Type](#model-type)
- [3. Intended Use](#intended-use)
  - [3.1. Intended Primary Users](#intended-primary-users)
  - [3.2. Intend Primary Uses](#intend-primary-uses)
- [4. Out-of-Scope Uses](#out-of-scope-uses)
- [5. Limitations](#limitations) 
- [6. Ethical Considerations](#ethical-considerations)
- [7. Carbon Footprint Evaluation](#carbon-footprint-evaluation)
- [8. Metrics](#metrics)
- [9. Training and Evaluation Data](#training-and-evaluation-data)
- [10. Quantitative Analysis](#quantitative-analysis)
- [11. Suported Inputs](#suported-inputs)
- [12. Table Summary](#table-summary)

## Model Date and Version

- May 2023
- Version 1.0

## Model Type
The model consists on a RoBERTa encoder pre-trained on a large tweet sentiment analysis corpus. The input data is processed by the backbone and passed into a simple one-layer classifier that will output the binary classification label. The model is fine-tuned with the Veiled Toxicity dataset.

## Intended Use

Input: English texts with a length of 128 characters.

### Intended Primary Users

Online forum or social media owners.

### Intend Primary Uses

Obtain a binary classification about the toxicity of an online comment. 

## Out-of-Scope Uses

Snetiment analysis of brief english texts of general content (not only possibly toxic comments)

## Limitations

The quality of user's input may impact the accuracy of the results. The factors that may detriment accuracy are:

- Not natural language (i.e. synthetic language).
- Content text (topic): different from onlinecomments.
- Content text: very neutral comments.
- Content text: too short text (not rich from the sentiment point of view).

## Ethical Considerations

Sentiment analysis tools are increasingly used everywhere. As sentiments and emotions are nuclear issues in people's lives, they are full of ethical concerns. Indentifying sentiments can be used to improve people's lives and enhance security online, but it also carries the option of abusing of this information in order to manipulate or harm people in many ways. Any use intended to make an individual profit or avoidance of behaviour rules should be considered a misuse of this model.

## Metrics

The metric used to evaluate the performance of this model is **recall by class**, where the classes are veiled toxicity, positive comments and overt toxicity.

The idea behind this is that we are not that concerned about precision by class, that is, how accurate are we when performing classification in that class, but how many examples are we classifying as toxic, even if the byproduct is the appearance of false positives (positive examples that are classified as toxic). 

Since veiled toxicity is hard to detect and often misclassified due the inherent nature of it, we primed the ability of the models to detect it in any way, even if that degrades the performance in other classes.

## Training and Evaluation Data

We used the split of the data proposed by the authors of the dataset, Han & Tsvetkov (2020). The training dataset is composed by 12000 examples divided into 2000 for veiled toxicity, 8000 for the positive comments, and 2000 for overt toxicity. The test set is composed of 1000 observations for each class.

## Quantitative Analysis

| Model                       | Dataset                 | Veiled toxicity recall | Positive examples recall | Overt toxicity recall |
|-----------------------------|-------------------------|------------------------|--------------------------|-----------------------|
| RoBERTa pre-trained encoder | Veiled Toxicity dataset |         80.86%         |          94.92%          |         99.42%        |

## Suported Inputs

Any string input is accepted. The text will be pre-processed automatically in order to limit the length of the input to 128 characters, with padding if necessary.

- tags:
text-sentiment-analysis
natural-language-processing
- datasets:
Veiled Toxicity dataset
- metrics:
Recall by class