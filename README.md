# Transformer-Language-Model-pre-training

## Overview

This is a personal side project where I dive into the world of transformer-based language models using the Hugging Face library. I built and trained a causal language model (similar to GPT) from scratch, pre-training it on Wikipedia articles and then fine-tuning it for an intent-classification task.

## Features

- **Pre-training**: Trained a GPT-2-like model on the WikiText-2 dataset to predict text sequences, evaluated with perplexity scores.
- **Fine-tuning**: Adapted the pre-trained model for intent classification using the IMDB dataset, measured by accuracy and F1 scores.
- **Hyperparameter Playground**: Tested various settings (batch size, learning rate, etc.) to find the sweet spot for performance.
- **Data Prep**: Processed datasets with tokenization, padding, and label encoding to fit the models’ needs.

## How It Works

### Datasets
- **WikiText-2 (Pre-training)**: A collection of Wikipedia articles with:
  - 36,718 training samples
  - 3,760 validation samples
  - 4,358 test samples
- **IMDB (Fine-tuning)**: 2,254 utterances with IOB slot tags and core relations (e.g., "who plays Luke in Star Wars" → movie.starring.actor).

### Data Pre-processing
I cleaned up the data by:
- Filling empty entries with "none".
- Tokenizing text and replacing underscores with dashes.
- Adding padding and unknown tokens.
- Binarizing labels for classification.

### Models
- **Pre-training**: Used `GPT2LMHeadModel` for causal language modeling.
- **Fine-tuning**: Switched to `GPT2ForSequenceClassification`, loading pre-trained weights and tweaking for classification.

### Experiments
I ran a bunch of tests to see what works best:
- **Pre-training**: Measured perplexity (lower is better) across different batch sizes, learning rates, attention heads, layers, and hidden dimensions.
- **Fine-tuning**: Checked accuracy and F1 scores with variations in batch size, learning rate, and threshold values.

## Technical Details
- **Language**: Python
- **Libraries**: Hugging Face Transformers, Datasets, PyTorch
- **Models**: GPT-2 variants (`GPT2LMHeadModel`, `GPT2ForSequenceClassification`)
- **Datasets**: WikiText-2, IMDB

## Key Findings

### Pre-training Highlights
- **Batch Size**: Smaller batches (e.g., 2) gave better perplexity (10.7) but took longer (8 mins) vs. larger batches (e.g., 64, 20 perplexity, 4 mins).
- **Learning Rate**: 1e-3 hit the best perplexity (8.9), while 1e-10 tanked it (28,875—barely learned anything!).
- **Attention Heads**: 2, 8, or 16 heads all scored ~10.7—dataset might not need the extra complexity.
- **Layers**: More layers (8 vs. 1) slightly improved perplexity (10.2 vs. 10.7) but doubled training time.
- **Hidden Dimension**: Bigger dimensions (1024) dropped perplexity to 8.5—more capacity, better results.

### Fine-tuning Highlights
- **Batch Size**: 2, 8, or 64 had similar accuracy (~0.83) and F1 (~0.88), but smaller batches took way longer (180s vs. 10s).
- **Learning Rate**: 1e-3 edged out with 0.85 accuracy and 0.9 F1—kept it default to avoid overfitting.
- **Threshold**: 0.01, 0.2, or 0.7 barely moved the needle—model’s already near its peak.

### Best Model
- **Pre-training**: Learning rate 5e-5, batch size 2, 8 heads, 1 layer, 1024 hidden dim → 8.5 perplexity.
- **Fine-tuning**: Learning rate 5e-5, batch size 8, threshold 0.2 → 0.85 accuracy, 0.9 F1.
- **Takeaway**: Pre-training gains didn’t fully carry over to fine-tuning—maybe the datasets are too different, or the model hit its limit.

## Project Showcases
Here’s a table summarizing some key results from my experiments:

| Phase         | Configuration                          | Perplexity | Accuracy | F1 Score | Training Time |
|---------------|----------------------------------------|------------|----------|----------|---------------|
| Pre-training  | Batch Size 2, LR 5e-5, 1024 Hidden Dim | 8.5        | -        | -        | 14 mins       |
| Pre-training  | Batch Size 64, LR 1e-3, 128 Hidden Dim | 20         | -        | -        | 4 mins        |
| Fine-tuning   | Batch Size 8, LR 1e-3, Threshold 0.2  | -          | 0.85     | 0.9      | 100 secs      |
| Fine-tuning   | Batch Size 64, LR 1e-10, Threshold 0.7| -          | 0.82     | 0.87     | 10 secs       |

*Notes: Pre-training focused on perplexity (lower is better), while fine-tuning targeted accuracy and F1 scores. The best setup balanced performance and training time.*
## Why I Built This
I wanted to get hands-on with transformers, play with Hugging Face’s tools, and see how far I could push a GPT-style model. It’s been a cool way to sharpen my Python skills, dig into NLP, and figure out what makes these models tick.

## Next Steps
- Try bigger datasets or more complex tasks.
- Experiment with other transformer architectures.
- Optimize training speed without sacrificing performance.
