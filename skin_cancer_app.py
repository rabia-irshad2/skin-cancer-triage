import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gradio as gr
from PIL import Image
import cohere
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# 1. Load Environment Variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

# 2. Optimized Data Pipeline Helpers
def load_and_preprocess_data(csv_path, img_dir):
    """Load CSV, verify image existence on disk, and split data safely"""
    df = pd.read_csv(csv_path)
    
    # Vectorized path creation
    df['img_filename'] = df['image_name'] + '.jpg'
    df['full_path'] = df['img_filename'].apply(lambda x: os.path.join(img_dir, x))
    
    # Verify file existence dynamically
    print("Verifying image files on disk...")
    df = df[df['full_path'].apply(os.path.exists)].reset_index(drop=True)
    
    if len(df) == 0:
        return None, None
        
    print(f"Found {len(df)} matching images on disk.")
    print(f"Class distribution - Benign: {sum(df['target']==0)}, Malignant: {sum(df['target']==1)}")
    
    # Stratified Train/Validation split (80% Train, 20% Val)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['target']
    )
    return train_df, val_df

def parse_image_and_label(filename, label):
    """Dynamically reads, decodes, and normalizes images from disk on-the-fly"""
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # Min-Max Normalization
    return image, label


# 3. CNN Architecture
def build_cnn_model():
    """Build a robust CNN model for binary classification"""
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # 0 = Benign, 1 = Malignant
    ])
    return model


# 4. Cohere LLM Triage Generation
def generate_triage_note(confidence, prediction):
    """Generate a medical triage note using Cohere API"""
    if not COHERE_API_KEY:
        return "⚠️ Cohere API key not configured. Please add COHERE_API_KEY to your .env file."
    
    diagnosis = "MALIGNANT" if prediction == 1 else "BENIGN"
    urgency = "HIGH - Requires immediate specialist referral" if prediction == 1 else "LOW - Routine follow-up recommended"
    
    prompt = f"""You are a dermatology AI assistant. Generate a concise triage note for an attending nurse.

DIAGNOSIS: {diagnosis}
Confidence: {confidence:.1f}%
URGENCY: {urgency}

Please include:
1. Visual features commonly seen in {diagnosis.lower()} lesions.
2. Urgency level and clinical timeframe for action.
3. Recommended next steps for the nurse to perform in the clinic.

Keep the response under 200 words, highly professional, and clinically actionable."""

    try:
        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=300,
            temperature=0.3
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"⚠️ Error generating triage note via API: {str(e)}\n\nManual assessment required."


# 5. Inference Pipeline for Gradio
def predict_and_triage(model, image):
    """Gradio prediction function wrapper"""
    try:
        if image is None:
            return "No image uploaded.", "Please upload an image first."
            
        # Convert image input to array format matching requirements
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = Image.fromarray(image.astype('uint8')).convert('RGB')
        
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run inference
        prediction = model.predict(img_array, verbose=0)
        confidence = prediction[0][0]
        
        predicted_class = 1 if confidence > 0.5 else 0
        confidence_percent = confidence * 100 if predicted_class == 1 else (1 - confidence) * 100
        
        # Markdown Formatting
        result_text = f"### **Prediction:** {'🔴 MALIGNANT' if predicted_class == 1 else '🟢 BENIGN'}\n"
        result_text += f"### **Confidence:** {confidence_percent:.1f}%\n"
        result_text += f"### **Urgency Status:** {'🚨 HIGH URGENCY - Immediate Action' if predicted_class == 1 else '📅 LOW URGENCY - Routine Tracking'}"
        
        triage_note = generate_triage_note(confidence_percent, predicted_class)
        return result_text, triage_note
    
    except Exception as e:
        return f"Error processing image: {str(e)}", "Inference engine failed."


# 6. Model Training Protocol
def train_and_save_model():
    """Builds and trains a model using streaming batch datasets to prevent RAM crashes."""
    csv_path = 'train_concat.csv'
    img_dir = 'train'
    
    train_df, val_df = load_and_preprocess_data(csv_path, img_dir)
    if train_df is None:
        print("Fatal Error: Missing assets. Check folder setup.")
        return None
        
    print(f"Preparing datasets (Train size: {len(train_df)}, Val size: {len(val_df)})")
    
    # Building Optimized tf.data Pipelines
    train_dataset = tf.data.Dataset.from_tensor_slices((train_df['full_path'].values, train_df['target'].values))
    train_dataset = train_dataset.shuffle(buffer_size=2000).map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_df['full_path'].values, val_df['target'].values))
    val_dataset = val_dataset.map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    model = build_cnn_model()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]
    
    print("\nStarting model training loops...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the model to file
    model.save('skin_cancer_model.h5')
    print("\n🎉 Model trained successfully and saved as 'skin_cancer_model.h5'")
    return model


# 7. Main Application Loop
def main():
    if os.path.exists('skin_cancer_model.h5'):
        print("Existing weights found! Loading 'skin_cancer_model.h5'...")
        model = keras.models.load_model('skin_cancer_model.h5')
    else:
        print("No existing model found. Initializing automated training pipeline...")
        model = train_and_save_model()
        if model is None:
            print("Failed to initialize system architecture. Terminating execution loop.")
            return
            
    print("Launching interface modules...")
    
    # Gradio Layout Configuration
    with gr.Blocks(title="DermAI Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🩺 DermAI: Skin Cancer Triage Assistant
        ### AI-powered open-ended laboratory triage workspace for under-resourced clinics.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Dermoscopic Target Input", type="numpy")
                submit_btn = gr.Button("🔍 Run Analysis Pipeline", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                result_output = gr.Markdown(label="System Classifications")
                triage_output = gr.Markdown(label="📋 Generated Nurse Triage Document")
        
        submit_btn.click(
            fn=lambda img: predict_and_triage(model, img),
            inputs=input_image,
            outputs=[result_output, triage_output]
        )
        
    demo.launch(share=False)

if __name__ == "__main__":
    main()
