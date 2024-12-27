# Emotion Detection with Logistic Regression and BERT

## Project Overview
This project focuses on detecting six different emotions from text data. The process involves two main stages:

1. **Training and evaluating a Logistic Regression model** on labeled emotion data.
2. **Fine-tuning a pretrained BERT model** for the emotion detection task using `Trainer` and `TrainingArguments` from the Hugging Face Transformers library.

## Dataset
- **Training Data:** Labeled text data for six emotions.
- **Validation Data:** Used for tuning the Logistic Regression model.
- **Testing Data:** Used to evaluate the fine-tuned BERT model.

## Project Workflow
1. **Train Logistic Regression Model:**
   - Preprocess the train, val, train text data to labels and sentences.
   - Vectorize each dataset into tf-idf vectors.
   - Train a Logistic Regression classifier to detect emotions.
   - Evaluate the model on validation data.
   - Evaluate the model on the testing data.
   - See `emotion-detection-regression-model.ipynb` for detailed implementation.

2. **Fine-Tune BERT Model:**
   - Load a pretrained BERT model.
   - Preprocess the train, val, train text data to labels and sentences.
   - Tokenize the sentences to be used for fine-tuning the BERT model.
   - Fine-tune the model on the emotion detection task using the training and validation data.
   - Evaluate the model on testing data.
   - Use `Trainer` and `TrainingArguments` for efficient model training.
   - See `Emotion_Detection_BERT (1).ipynb` for detailed implementation.

3. **Re-evaluation:**
   - Load the saved BERT model.
   - Re-evaluate on testing data.
   - Record class-wise accuracy and results.
   - Refer to `Emotion_Detection_Reuse_Model.ipynb` for the complete re-evaluation process.

## Installation
```bash
# Clone the repository
git clone https://github.com/username/emotion-detection-nlp.git
cd emotion-detection-nlp

# Install required dependencies
pip install -r requirements.txt
```

## Results
- **Logistic Regression Model Accuracy:** 85%
- **Fine-Tuned BERT Model Accuracy:** 92%
- **Class-wise Accuracy (BERT):**
  - Joy: 95.83%
  - Sadness: 97.25%
  - Anger: 89.09%
  - Love: 81.13%
  - Fear: 89.29%
  - Surprise: 74.23%

## Conclusion
The emotions that weren’t as accurate were the ‘surprise’, ‘love’, ‘anger’, and ‘fear’ emotions. ‘Fear’ and ‘anger’ were still good with about 89% accuracy each, but ‘love’ and ‘surprise’ were not as accurate as we want with about 81% and 74% accuracy (see images after explanation). However, this corresponds with the distribution of the emotions in the training and validation sets.  ‘Love’ and ‘surprise’ each had the lowest distributions. So, this can point to the fact that since the datasets had lower examples for those emotions, the model wasn’t as accurate in those emotions. This shows that this particular problem can grow and get better. With more examples for ‘love’ and ‘surprise’, even for ‘fear’ and ‘anger’, the accuracy of the model can reach an accuracy of high 90s.

Additionally, looking through the sentences that were misclassified and the confusion matrix, we could see that some emotions can be mixed up. Here is one example: “I feel blessed to know this family”. The true label here is ‘love’, but the model predicted ‘joy’. Looking at it firsthand, it makes sense why the model would think this sentence correlates with ‘joy’, but also makes sense why it could be ‘love’. ‘Love’ and ‘joy’ tend to correlate with each other, so that could be a reason why the model mixed them up quite a bit. Now, with ‘fear’ and ‘surprise’, the sentence,  “I indicated then I was feeling quite overwhelmed with work responsibilities teaching traveling and writing”, was predicted as ‘surprise’ when it was actually ‘fear’. ‘Fear’ and ‘surprise’ tend to correspond as well because they both can happen in the same situation like getting scared where you can be surprised but feel that sense of fear. Also, ‘fear’ and ‘anger’ were mixed up a few times as well. In the sentence, “I know what you mean about feeling agitated”, the prediction was “anger”, but the real label was “fear”. The word “agitated” probably what tricked the model to choose “anger”, but the individual wasn’t angry, but just fearful about a time when they were angry.

What seems like a recurring problem is that there is not enough data for certain emotions. Also, more emotions can make it easier to break apart different emotions that could be difficult for a model to differentiate between or a dataset with combinations of emotions paired with a sentence rather than limiting to one emotion can make the results more meaningful and more accurate. 

## Dependencies
- Transformers
- Sklearn
- Pytorch
- Datasets
- Numpy
- Matplotlib
- Seaborn

## Contributing
Feel free to open an issue or submit a pull request if you would like to contribute to this project.

