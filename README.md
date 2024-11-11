# 🏨 Hotel Review Analyzer

A deep learning web application that predicts hotel review ratings based on user input using a neural network model.

## 🌟 Features
- **Predictive Analysis**: Enter hotel reviews and receive a predicted rating.
- **Real-time Validation**: Character count validation ensures input meets minimum requirements.
- **Modern UI**: Sleek, responsive design with smooth transitions and hover effects.
- **Confidence Display**: Shows the confidence level of predictions.

## 🏨 Supported Rating Scale
- ⭐ 1 Star
- ⭐⭐ 2 Stars
- ⭐⭐⭐ 3 Stars
- ⭐⭐⭐⭐ 4 Stars
- ⭐⭐⭐⭐⭐ 5 Stars

## 🛠️ Technology Stack
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask
- **Machine Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy

## 📋 Prerequisites
- Python 3.8+
- pip package manager

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/hotel-review-analyzer.git
   cd hotel-review-analyzer
   ```

2. **Install Required Packages**:
   ```bash
   pip install flask tensorflow pandas numpy
   pip install -r requirements.txt
   ```

3. **Ensure Model Files**:
   - `hotel_review_model.h5` and `tokenizer.pickle` should be in the `models` directory.

## 🚀 Running the Application
1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser and navigate to**:
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

- hotel-review-analyzer/
- ├── app.py             # Flask application
- ├── model.py           # Model training code
- ├── static/
- │   ├── style.css      # Styling
- │   └── script.js      # Frontend logic
- ├── templates/
- │   └── index.html     # Main page
- └── models/
-    ├── hotel_review_model.h5 # Trained model
-    └── tokenizer.pickle      # Tokenizer for text processing

## 🤖 Model Architecture
The project uses a neural network model with the following architecture:

```python
def create_model(vocab_size):
    """Create a simplified neural network model with basic CNN and LSTM paths."""
    
    # Input layer
    input_layer = Input(shape=(MAX_LENGTH,))
    
    # Embedding layer
    embedding = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LENGTH)(input_layer)
    
    # Single CNN path
    conv = Conv1D(64, 5, activation='relu', padding='same')(embedding)
    conv = BatchNormalization()(conv)
    conv = MaxPooling1D(2)(conv)
    conv = GlobalAveragePooling1D()(conv)
    
    # Simple LSTM path
    lstm = Bidirectional(LSTM(32, return_sequences=True))(embedding)
    lstm = GlobalAveragePooling1D()(lstm)
    
    # Merge paths
    merged = Concatenate()([conv, lstm])
    
    # Simplified dense layers
    dense = Dense(128, activation='relu')(merged)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)
    
    # Output layer
    output_layer = Dense(5, activation='softmax')(dense)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model
```

## 🎨 UI Features
- **Frosted Glass Card Design**: Modern aesthetic with a clean look.
- **Responsive Layout**: Adapts to different screen sizes.
- **Real-time Character Count**: Ensures input meets minimum length.
- **Smooth Transitions**: Enhances user experience with animations.

---
![Good Review](good_review.png)
![Bad Review](bad_review.png)
Made with 💙 by Peter Shamoun





