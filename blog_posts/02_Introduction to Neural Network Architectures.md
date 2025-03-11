---
title: Introduction to Neural Network Architectures
date: 2025-02-15
author: Heuijee Yun
excerpt: Overview of modern neural network architectures including CNNs, RNNs, Transformers, and recent advancements like Vision Transformers (ViT) and Graph Neural Networks (GNNs).
---

Neural networks have revolutionized machine learning across numerous domains. This post explores key architectures that have driven recent advances.

## Convolutional Neural Networks (CNNs)

CNNs have transformed computer vision through their ability to automatically learn hierarchical features from images:

```python
import tensorflow as tf

def create_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model
```

## Recurrent Neural Networks (RNNs)

RNNs and their variants (LSTM, GRU) excel at sequence modeling tasks like natural language processing:

![RNN Structure](images/rnn_diagram.jpg)

Key components of an LSTM cell include:
- Input gate
- Forget gate
- Output gate
- Cell state

## Transformer Architecture

Transformers have become the dominant paradigm for NLP tasks:

```python
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
```

## Recent Advances

### Vision Transformers (ViT)

Vision Transformers have challenged CNNs for image recognition tasks by applying transformer-based attention to image patches.

### Graph Neural Networks (GNNs)

GNNs process data represented as graphs, enabling applications in:
- Social network analysis
- Molecular property prediction
- Recommendation systems

## Conclusion

The evolution of neural network architectures continues to drive progress in artificial intelligence, enabling increasingly sophisticated applications across domains.

## References

1. LeCun, Y., et al. (2024). "Deep Learning: Past, Present, and Future"
2. Vaswani, A., et al. (2017). "Attention Is All You Need" 