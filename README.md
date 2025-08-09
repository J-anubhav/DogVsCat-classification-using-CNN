# 🐕🐱 Dog vs Cat Classification using Enhanced CNN

A deep learning project that classifies images of dogs and cats using Convolutional Neural Networks (CNN) with transfer learning and ensemble methods.

## 📊 Project Results

| Model | Accuracy | Improvement |
|-------|----------|-------------|
| Original CNN | 72.75% | Baseline |
| Enhanced CNN | 75.50% | +2.75% |
| Transfer Learning (VGG16) | 94.50% | +21.75% |
| **Ensemble** | **~95%** | **+22.25%** |

## 🚀 Features

- **Enhanced CNN Architecture**: Improved model with batch normalization and dropout layers
- **Transfer Learning**: VGG16 pre-trained model for superior feature extraction
- **Advanced Data Augmentation**: Rotation, shifting, zooming, and brightness variations
- **Ensemble Methods**: Combines multiple models for maximum accuracy
- **Comprehensive Evaluation**: Confusion matrices, classification reports, and visual predictions
- **Learning Rate Scheduling**: Adaptive learning rate with plateau reduction
- **Early Stopping**: Prevents overfitting with patience-based stopping

## 📁 Project Structure

```
DogVsCat-classification-using-CNN/
│
├── input.csv              # Training image data (flattened)
├── input_test.csv         # Test image data (flattened)
├── labels.csv             # Training labels (0=Dog, 1=Cat)
├── labels_test.csv        # Test labels (0=Dog, 1=Cat)
├── enhanced_model.py      # Main enhanced model implementation
├── step_by_step.py        # Step-by-step implementation guide
├── improved_cnn_best.h5   # Best improved CNN model weights
├── transfer_learning_best.h5  # Best transfer learning model weights
└── README.md              # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd DogVsCat-classification-using-CNN
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
   ```

## 💾 Data Format

The dataset consists of 100x100 RGB images flattened into CSV format:

- **Training Data**: 2,000 images (1,000 dogs + 1,000 cats)
- **Test Data**: 400 images (200 dogs + 200 cats)
- **Image Shape**: 100×100×3 (30,000 features per image)
- **Labels**: Binary (0 = Dog, 1 = Cat)

## 🧠 Model Architectures

### 1. Enhanced CNN
```
Input (100x100x3)
├── Conv2D(32) + BatchNorm + Conv2D(32) + MaxPool + Dropout
├── Conv2D(64) + BatchNorm + Conv2D(64) + MaxPool + Dropout
├── Conv2D(128) + BatchNorm + Conv2D(128) + MaxPool + Dropout
├── Conv2D(256) + BatchNorm + Dropout
├── GlobalAveragePooling2D
├── Dense(512) + BatchNorm + Dropout
├── Dense(256) + Dropout
└── Dense(1, sigmoid)
```
**Parameters**: 849,313

### 2. Transfer Learning (VGG16)
```
VGG16 Base (frozen early layers)
├── GlobalAveragePooling2D
├── Dense(512) + BatchNorm + Dropout
├── Dense(256) + Dropout
└── Dense(1, sigmoid)
```
**Parameters**: 15,110,977

## 🏃‍♂️ Usage

### Quick Start
```python
# Run the complete enhanced pipeline
python enhanced_model.py
```

### Step-by-Step Execution
```python
# For detailed step-by-step implementation
python step_by_step.py
```

### Custom Training
```python
from enhanced_model import *

# Load and preprocess data
X_train, X_test, Y_train, Y_test = load_and_preprocess_data()

# Create and train improved model
model = create_improved_model()
history = train_model(model, train_gen, val_gen)

# Evaluate model
accuracy = evaluate_and_visualize(model, X_test, Y_test)
```

## 📈 Training Configuration

### Data Augmentation
- **Rotation**: ±30 degrees
- **Width/Height Shift**: ±30%
- **Horizontal Flip**: Enabled
- **Zoom**: ±30%
- **Shear**: ±20%
- **Brightness**: 80%-120%

### Training Parameters
- **Batch Size**: 32
- **Initial Learning Rate**: 0.001 (CNN), 0.0001 (Transfer Learning)
- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Loss Function**: Binary Crossentropy
- **Early Stopping**: Patience = 10 epochs
- **Learning Rate Reduction**: Factor = 0.5, Patience = 5 epochs

## 📊 Performance Analysis

### Confusion Matrix Comparison

**Original Model (72.75%)**
```
          Predicted
Actual   Dog  Cat
Dog      146   54
Cat       55  145
```

**Enhanced CNN (75.50%)**
```
          Predicted
Actual   Dog  Cat
Dog      133   67
Cat       31  169
```

**Transfer Learning (94.50%)**
```
          Predicted
Actual   Dog  Cat
Dog      183   17
Cat        5  195
```

### Key Improvements

1. **Architecture Enhancement**: +2.75% accuracy
   - Added batch normalization for faster convergence
   - Implemented GlobalAveragePooling2D to reduce overfitting
   - Strategic dropout placement

2. **Transfer Learning**: +21.75% accuracy
   - Pre-trained VGG16 features
   - Fine-tuning approach
   - Significantly better feature extraction

3. **Ensemble Method**: +22.25% total improvement
   - Combines predictions from multiple models
   - Reduces prediction variance
   - Maximum achievable accuracy

## 🎯 Advanced Features

### Model Callbacks
- **Early Stopping**: Monitors validation accuracy with patience
- **Learning Rate Scheduler**: Reduces LR when validation loss plateaus
- **Model Checkpointing**: Saves best model weights automatically

### Visualization Tools
- Training history plots (accuracy and loss)
- Confusion matrices with heatmaps
- Prediction examples with confidence scores
- Per-class performance analysis

### Ensemble Methods
```python
# Create ensemble prediction
ensemble_pred = create_ensemble_prediction([model1, model2], X_test)
```

## 🚀 Future Improvements

1. **Advanced Architectures**
   - ResNet, EfficientNet, Vision Transformers
   - Custom attention mechanisms

2. **Data Enhancement**
   - Larger dataset collection
   - Advanced augmentation techniques (CutMix, MixUp)
   - Synthetic data generation

3. **Optimization Techniques**
   - Hyperparameter tuning with Optuna
   - Learning rate schedules (Cosine Annealing)
   - Advanced regularization techniques

## 🔧 Troubleshooting

### Common Issues

1. **Memory Errors**
   ```python
   # Reduce batch size
   batch_size = 16  # instead of 32
   ```

2. **Overfitting**
   ```python
   # Increase dropout rates
   Dropout(0.6)  # instead of 0.5
   ```

3. **Slow Training**
   ```python
   # Use mixed precision
   tf.keras.mixed_precision.set_global_policy('mixed_float16')
   ```

## 📝 Requirements

```
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **VGG16 Architecture**: Simonyan & Zisserman (2014)
- **Transfer Learning**: Pre-trained ImageNet weights
- **Dataset**: Custom dog and cat image collection
- **Framework**: TensorFlow/Keras ecosystem

## 📞 Contact

For questions, suggestions, or collaboration:
- **Email**: your.email@example.com
- **GitHub**: [@J-anubhav](https://github.com/J-anubhav)
- **LinkedIn**: [Anubhav Jha](https://linkedin.com/in/jha_anubhav)

---

**🎉 Achieved 94.50% accuracy with transfer learning - a 21.75% improvement over the baseline model!**