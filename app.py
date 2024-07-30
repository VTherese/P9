import streamlit as st
import pandas as pd
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, ops
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go

# Définir le thème clair et la mise en page large
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Proof of Concept YOLOv9", page_icon="📃")

# Load the YOLO model
model = YOLO('modele.pt')

# Function to load images
def load_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            img = Image.open(os.path.join(image_folder, filename))
            if img is not None:
                images.append((filename, img))
    return images

# Load images and data
def get_images_by_label(label):
    folder_map = {
        'Minor': 'train/minor',
        'Moderate': 'train/moderate',
        'Severe': 'train/severe',
        'Multi': 'train/multi'
    }
    if label in folder_map:
        return load_images(folder_map[label])
    else:
        images = []
        for lbl in folder_map.values():
            images.extend(load_images(lbl))
        return images

# Sidebar navigation
st.sidebar.markdown("<h1 style='font-size: 20px;'>📃 Proof of Concept YOLOv9</h1>", unsafe_allow_html=True)
section = st.sidebar.radio("Menu", ["Analyse Exploratoire", "Modèle"])

# Function to display exploratory data analysis
def exploratory_data_analysis():
    st.header("🔍 Analyse Exploratoire des Données")
    
    # Create two columns
    col1, col2 = st.columns(2)

    with col2:
        # Image label selection
        st.subheader(f"🖼️ 6479 Images")
        st.write(f"""<p style='font-size: 20px;'>
            Entrainement : 5823 | Validation : 408 | Test : 248 <br>
        </p>""", unsafe_allow_html=True)
        st.write(f"<p style='font-size: 16px; margin-bottom: 0px;'> Choisissez un label d'image à afficher </p> ", unsafe_allow_html=True)
        label_choice = st.selectbox("", ["Minor", "Moderate", "Severe", "Multi"])
        images = get_images_by_label(label_choice)
            
        # Display image thumbnails in a grid of 2 columns
        for i in range(0, len(images[:4]), 2):
            cols = st.columns(2)
            for col, (filename, img) in zip(cols, images[i:i+2]):
                col.image(img, width=260)

    with col1:
        st.subheader("📁 Source")
        st.write(f"""<p style='font-size: 20px;'>
             Dataset car_damage_severity sur <a href='https://universe.roboflow.com/pob/car_damage_severity-cnfnt/dataset/4' target='_blank'>Roboflow</a>
        </p>""", unsafe_allow_html=True)
        st.subheader("📊 3 Classes")
        # Class balance data
        class_counts = {
            'Moderate': 2466,
            'Minor': 1494,
            'Severe': 676
        }
        # Pie chart for class balance with Plotly
        fig = px.pie(values=class_counts.values(), names=class_counts.keys(), 
                     color_discrete_sequence=['skyblue', 'lightgreen', 'red'], 
                     title="Répartition des Classes")
        fig.update_traces(textinfo='percent+label', textfont_size=20)  # Update font size for percentages
        fig.update_layout(legend=dict(font=dict(size=20)),  # Update font size for legend
                          title=dict(text="Répartition des Classes", font=dict(size=20),),
                          width=600,  # Reduce the width of the chart
                          height=350,  # Reduce the height of the chart
                          margin=dict(l=10, r=10, t=40, b=10))  # Reduce margins 
        
        st.plotly_chart(fig)
        st.subheader("⚙️ Traitements & Augmentations")
        st.write(f"""<p style='font-size: 20px;'>
            &nbsp;&nbsp;&nbsp;&nbsp;➡️ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Redimensionnement : 640x640 <br>
            &nbsp;&nbsp;&nbsp;&nbsp;➡️ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Flip horizontal <br>
            &nbsp;&nbsp;&nbsp;&nbsp;➡️ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Luminosité +/- 25% <br>
            &nbsp;&nbsp;&nbsp;&nbsp;➡️ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Exposition +/- 25%
        </p>""", unsafe_allow_html=True)

# Function to display prediction model section
def prediction_model():
    st.header(" 🖥️ Modèle ")

    col1, col2 = st.columns(2)

    # Model performance summary
    with col1:
        st.subheader("📊 Performances")
        
        # Performance data for different classes
        performance_data = {
            'all': [0.68, 0.728, 0.713, 0.625],
            'minor': [0.693, 0.635, 0.605, 0.485],
            'moderate': [0.658, 0.728, 0.697, 0.622],
            'severe': [0.689, 0.821, 0.836, 0.768]
        }
        metrics = ["Précision (P)", "Rappel (R)", "mAP50", "mAP50-95"]

        # Selectbox for class choice
        st.write(f"<p style='font-size: 16px; margin-bottom: 0px;'> Choisissez une classe pour voir les performances associées </p> ", unsafe_allow_html=True)
        class_choice = st.selectbox("", ["all", "minor", "moderate", "severe"])
        performance_scores = performance_data[class_choice]

        # Define the title based on the class choice
        if class_choice == "all":
            title = "Performances moyennes (tous types de dommages)"
        else:
            title = f"Performances pour les dommages de type : {class_choice}"

        # Create a bar chart with Plotly
        fig = go.Figure(data=[
            go.Bar(name=metrics[0], x=[class_choice], y=[performance_scores[0]], marker_color='blue', text=[f'{performance_scores[0]*100:.1f}%'], textposition='auto'),
            go.Bar(name=metrics[1], x=[class_choice], y=[performance_scores[1]], marker_color='orange', text=[f'{performance_scores[1]*100:.1f}%'], textposition='auto'),
            go.Bar(name=metrics[2], x=[class_choice], y=[performance_scores[2]], marker_color='green', text=[f'{performance_scores[2]*100:.1f}%'], textposition='auto'),
            go.Bar(name=metrics[3], x=[class_choice], y=[performance_scores[3]], marker_color='red', text=[f'{performance_scores[3]*100:.1f}%'], textposition='auto')
        ])
        # Update the layout
        fig.update_layout(
            barmode='group', 
            xaxis_title="Classe", 
            yaxis_title="Score", 
            title=title,
            title_font_size=20,  # Increase title font size
            legend_font_size=20,  # Increase legend font size
            yaxis=dict(range=[0, 1]),  # Ensure the y-axis goes to 1
            width=400,  # Reduce the width of the chart
            height=250,  # Reduce the height of the chart
            margin=dict(l=10, r=10, t=30, b=10)  # Reduce margins
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🖼️ Choisissez une image à prédire ")
        # File uploader for the user to upload an image
        st.write(f"<p style='font-size: 16px; margin-bottom: 0px;'> Téléchargez une image </p> ", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            original_width, original_height = image.size
            # Display the uploaded image
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(np.array(image))
            ax.axis('off')
            st.pyplot(fig)
        

    with col2:
        st.subheader("🎚️ Paramétrage des seuils")
        st.write(f"<p style='font-size: 20px; margin-bottom: 0px;'> 💡 Seuil de confiance. </p> ", unsafe_allow_html=True)
        st.write(f"<p style='font-size: 16px;'> La confiance représente la probabilité qu'un objet détecté appartienne à une classe spécifique. En choisissant un seuil, il est possible de garder les prédictions les plus probables et d'éliminer celles qui sont moins sûres. </p> ", unsafe_allow_html=True)
        confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5)
        st.write(f"<p style='font-size: 20px; margin-bottom: 0px;'> 💡 Seuil de chevauchement. </p> ", unsafe_allow_html=True)
        
        st.write(f"<p style='font-size: 16px;'> Lorsqu'un objet est détecté plusieurs fois avec des boîtes se chevauchant au delà d'un certain seuil, seule la boîte avec le score de confiance le plus élevé est retenue. </p> ", unsafe_allow_html=True)
        
        overlap_threshold = st.slider("Seuil de chevauchement (IoU)", 0.0, 1.0, 0.5)
        st.write(f"""
            <p style='font-size: 20px;'> 🔦 Objets affichés :<br>
            <div style='margin-left: 140px;font-size: 20px;'>➡️ Confiance > {confidence_threshold}</div>
            <div style='margin-left: 140px;font-size: 20px;'>➡️ Chevauchement < {overlap_threshold}</div>
            </p>
            """, unsafe_allow_html=True)


        
        if uploaded_file is not None:
            st.subheader("🔍 Résultat de prédiction ")
            # Prepare the image for the model
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0)

            # Perform prediction
            results = model(image_tensor)
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
            labels = results[0].boxes.cls.cpu().numpy()  # Predicted classes
            
            boxes_tensor = torch.tensor(boxes)
            scores_tensor = torch.tensor(scores)
            labels_tensor = torch.tensor(labels)

            # Apply Non-Maximum Suppression (NMS)
            keep_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold=overlap_threshold)
            nms_boxes = boxes_tensor[keep_indices]
            nms_scores = scores_tensor[keep_indices]
            nms_labels = labels_tensor[keep_indices]

            # Filter predictions by confidence score after applying NMS
            filtered_indices = nms_scores > confidence_threshold
            filtered_boxes = nms_boxes[filtered_indices]
            filtered_scores = nms_scores[filtered_indices]
            filtered_labels = nms_labels[filtered_indices]

            # Adjust coordinates of the boxes to original image size
            scale_x = original_width / 640
            scale_y = original_height / 640
            filtered_boxes[:, [0, 2]] *= scale_x
            filtered_boxes[:, [1, 3]] *= scale_y
            # Clip the boxes to be within the image boundaries
            filtered_boxes[:, [0, 2]] = np.clip(filtered_boxes[:, [0, 2]], 0, original_width)
            filtered_boxes[:, [1, 3]] = np.clip(filtered_boxes[:, [1, 3]], 0, original_height)
            # Define colors for each class
            colors = {
                'minor': 'green',
                'moderate': 'blue',
                'severe': 'red'
            }

            # Display the bounding boxes and labels on the image
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(np.array(image))
            for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
                class_name = model.names[int(label)]
                color = colors.get(class_name, 'yellow')  #  if class_name not found
                ax.add_patch(plt.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    fill=False, edgecolor=color, linewidth=2
                ))
                ax.text(box[0], box[1] - 10, f'{class_name} {score:.2f}',
                        bbox=dict(facecolor=color, alpha=0.5), fontsize=18, color='white')
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            st.pyplot(fig)

# Display the selected section
if section == "Analyse Exploratoire":
    exploratory_data_analysis()
elif section == "Modèle":
    prediction_model()
