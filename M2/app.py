# -*- coding: utf-8 -*-
"""
CIFAR-10 Image Classification Web App using Streamlit
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent

sys.path.insert(0, str(BASE_DIR))

from models.model_M2 import M1_CNN

# ============================================================================
# CIFAR-10 CLASS NAMES
# ============================================================================
CIFAR10_CLASSES = (
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
)

#===========================================================================
# Cifar 100
#===========================================================================
# CIFAR10_CLASSES = (
#     "Apple", "Aquarium Fish", "Baby", "Bear", "Beaver", "Bed", "Bee", "Beetle", 
#     "Bicycle", "Bottle", "Bowl", "Boy", "Bridge", "Bus", "Butterfly", "Camel", 
#     "Can", "Castle", "Caterpillar", "Cattle", "Chair", "Chimpanzee", "Clock", 
#     "Cloud", "Cockroach", "Couch", "Crab", "Crocodile", "Cup", "Dinosaur", 
#     "Dolphin", "Elephant", "Flatfish", "Forest", "Fox", "Girl", "Hamster", 
#     "House", "Kangaroo", "Keyboard", "Lamp", "Lawn Mower", "Leopard", "Lion", 
#     "Lizard", "Lobster", "Man", "Maple Tree", "Motorcycle", "Mountain", "Mouse", 
#     "Mushroom", "Oak Tree", "Orange", "Orchid", "Otter", "Palm Tree", "Pear", 
#     "Pickup Truck", "Pine Tree", "Plain", "Plate", "Poppy", "Porcupine", 
#     "Possum", "Rabbit", "Raccoon", "Ray", "Road", "Rocket", "Rose", 
#     "Sea", "Seal", "Shark", "Shrew", "Skunk", "Skyscraper", "Snail", "Snake", 
#     "Spider", "Squirrel", "Streetcar", "Sunflower", "Sweet Pepper", "Table", 
#     "Tank", "Telephone", "Television", "Tiger", "Tractor", "Train", "Trout", 
#     "Tulip", "Turtle", "Wardrobe", "Whale", "Willow Tree", "Wolf", "Woman", 
#     "Worm"
# )
#===========================================================================
# 5 class animal: cat, dog, ELEPHANT, HORSE, LION
#===========================================================================
# CIFAR10_CLASSES = (
#     "Cat",
#     "Dog",
#     "Elephant",
#     "Horse",
#     "Lion"
# )

# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = str(BASE_DIR / "checkpoint(pth)" / "best_M2_CIFAR10_32.pth")


# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================
@st.cache_resource
def load_model(checkpoint_path=CHECKPOINT_PATH):
    """
    Load the pre-trained M1_CNN model from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file containing model weights.
        
    Returns:
        model (M1_CNN): Model loaded in eval mode on the appropriate device.
        
    Raises:
        FileNotFoundError: If checkpoint file is not found.
        Exception: If there's an error loading the checkpoint.
    """
    try:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
        
        model = M1_CNN(variant=IMG_SIZE, num_classes=len(CIFAR10_CLASSES))
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        
        # Extract model state dict from checkpoint (old pth format)
        # if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        #     model.load_state_dict(checkpoint['model_state_dict'])
        # elif isinstance(checkpoint, dict):
        #     model.load_state_dict(checkpoint)
        # else:
        #     model.load_state_dict(checkpoint.state_dict())

        # Extract model state dict from checkpoint (new pth)
        model_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_state_dict)
        
        model = model.to(DEVICE)
        model.eval()
        
        return model
    
    except FileNotFoundError as e:
        st.error(f"❌ Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
@st.cache_data
def get_preprocessing_transforms():
    """
    Get the image preprocessing transforms.
    
    Returns:
        transforms.Compose: Composition of preprocessing transforms.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for inference.
    
    Args:
        image (PIL.Image.Image): PIL Image object.
        
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, 32, 32).
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = get_preprocessing_transforms()
    image_tensor = transform(image)
    
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

# ============================================================================
# INFERENCE
# ============================================================================
def predict(model, image_tensor: torch.Tensor) -> tuple:
    """
    Perform inference on a preprocessed image tensor.
    
    Args:
        model (nn.Module): PyTorch model in eval mode.
        image_tensor (torch.Tensor): Preprocessed image tensor of shape (1, 3, 32, 32).
        
    Returns:
        tuple: (predicted_class_name, confidence_percentage)
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        
        logits = model(image_tensor)
        print(f"Logits: {logits.cpu().numpy()}")  
        
        probabilities = torch.softmax(logits, dim=1)
        print(f"Probabilities: {probabilities.cpu().numpy()}")  
        
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = CIFAR10_CLASSES[predicted_idx.item()]
        confidence_percentage = confidence.item() * 100
        
        return predicted_class, confidence_percentage, probabilities.cpu().numpy()[0]

# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    """Main Streamlit app function."""
    
    st.set_page_config(
        page_title="CIFAR-10 Classifier",
        page_icon="🖼️",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        .main {
            padding-top: 0rem;
        }
        .stMetric {
            background-color: rgba(255, 255, 255, 0.1); 
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.2); 
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🖼️ CIFAR-10 Image Classifier")
    st.markdown("""
    This web app classifies images into one of the **10 CIFAR-10 categories**.
    
    **Supported Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
    
    Upload an image file (PNG, JPG, or JPEG) and the model will predict its class with a confidence score.
    """)
    
    st.divider()
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            f"""
            **Model:** M1_CNN (CIFAR-10)
            
            **Input Size:** 32×32 pixels
            
            **Device:** {str(DEVICE).upper()}
            
            **Checkpoint:** `{CHECKPOINT_PATH}`
            """
        )
        st.markdown("---")
        st.subheader("💡 Tips")
        st.markdown(
            """
            • Upload clear, well-lit images
            • Crop images to focus on the main subject
            • Works best with images similar to CIFAR-10 dataset
            """
        )
    
    # Load model (cached)
    with st.spinner("Loading model..."):
        model = load_model(CHECKPOINT_PATH)
    st.success("✅ Model loaded successfully!")
    
    st.divider()
    
    uploaded_file = st.file_uploader(
        "📤 Upload an image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    # load image
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Uploaded Image")
                st.image(image, use_container_width=True, caption="Original image")
            
            # Preprocess and predict
            with st.spinner("Analyzing image..."):
                image_tensor = preprocess_image(image)
                predicted_class, confidence, probabilities = predict(model, image_tensor)
            
            # Display results
            with col2:
                st.subheader("🎯 Prediction Results")
                
                st.metric(
                    label="Predicted Class",
                    value=predicted_class,
                    delta=f"{confidence:.2f}% confidence"
                )
                
                st.divider()
                
                # Confidence score with progress bar
                st.markdown(f"### Confidence: **{confidence:.2f}%**")
                st.progress(confidence / 100.0)
                
            st.divider()
            
            st.subheader("📊 Class Probabilities")
            
            # Create probability display
            prob_df = pd.DataFrame({
                "Class": CIFAR10_CLASSES,
                "Probability": [f"{prob*100:.2f}%" for prob in probabilities],
                "Score": probabilities
            })
            
            # Sort by probability (descending)
            prob_df = prob_df.sort_values("Score", ascending=False).reset_index(drop=True)
            
            # Display as bar chart
            st.bar_chart(
                data=prob_df.set_index("Class")["Score"],
                use_container_width=True,
                height=300
            )
            
            # Display as table
            st.dataframe(
                prob_df[["Class", "Probability"]],
                use_container_width=True,
                hide_index=True
            )
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
    
    else:
        st.info("👆 Please upload an image to get started!")
        
        with st.expander("📚 Learn more about CIFAR-10"):
            st.markdown("""
            **CIFAR-10** is a dataset of 60,000 32×32 color images in 10 classes:
            
            1. **Airplane** - Commercial and private aircraft
            2. **Automobile** - Cars and vehicles
            3. **Bird** - Various bird species
            4. **Cat** - Domestic cats
            5. **Deer** - Deer and similar animals
            6. **Dog** - Domestic dogs
            7. **Frog** - Frogs and amphibians
            8. **Horse** - Horses and similar animals
            9. **Ship** - Ships and boats
            10. **Truck** - Trucks and heavy vehicles
            """)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import pandas as pd
    main()
