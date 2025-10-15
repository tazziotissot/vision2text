# vision2text: Image Captioning with Transfer Learning

> *Work in progress – code and results will be released soon.*

This project explores **automatic image captioning**, where an algorithm learns to generate textual descriptions of images.  
The model is trained on the **Flickr8k** dataset, which contains 8,000 images of everyday scenes, each annotated with multiple human-written captions.  

---

## Project Overview

The objective of this work is to build a **deep learning pipeline** capable of translating visual information into natural language.  
To achieve this, the project implements an **encoder–decoder architecture** combining **computer vision** and **natural language processing (NLP)** components.

---

### Encoder: Visual Feature Extraction
- Based on a **ResNet50** convolutional neural network pretrained on **ImageNet**  
- The convolutional base acts as a **feature extractor**, producing compact embeddings that capture high-level visual semantics  
- The final fully connected layers are replaced with a projection layer that maps image features to the same latent space as the word embeddings

This approach allows the model to leverage the generalization power of **transfer learning** while adapting to the captioning task.

---

### Decoder: Sequence Generation

- Implemented as a **Long Short-Term Memory (LSTM)** network  
- Initialized with **GloVe 6B 100D** pretrained embeddings to inject prior semantic knowledge into the word representation space  
- During training, **teacher forcing** is used to guide the sequence generation process by feeding the ground-truth word at each timestep instead of the model’s prediction  
- The decoder learns to predict the next word given the current visual context and the previously generated sequence

---

### Training & Evaluation

- Dataset: [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)  
- Loss: categorical cross-entropy on the predicted sequence  
- Optimizer: Adam  
- Metric: **BLEU score** to evaluate the similarity between generated and reference captions

The BLEU evaluation provides a quantitative measure of how close the generated descriptions are to human language in terms of structure and meaning.
