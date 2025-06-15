import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle
from Wide_ResNet_32_10 import Model
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

# Set page configuration
st.set_page_config(
    page_title="CIFAR-10 Adversarial Training Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .attack-info {
        background-color: #fff2cc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffa500;
    }
    .defense-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_cifar10_labels():
    """Load CIFAR-10 class labels"""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

@st.cache_resource
def load_model(model_path):
    """Load the trained CIFAR-10 model"""
    try:
        # Initialize model for CIFAR-10
        model = Model(mode="eval", dataset="cifar10")
        
        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            model.eval()
            return model, True
        else:
            st.warning(f"Model file not found: {model_path}")
            return model, False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

def unpickle(file):
    """Unpickle CIFAR data"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

@st.cache_data
def load_sample_images(num_samples=50):
    """Load sample images from CIFAR-10 dataset"""
    try:
        data_path = "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/cifar10/cifar-10-batches-py/test_batch"
        labels = load_cifar10_labels()
        
        if os.path.exists(data_path):
            test_dict = unpickle(data_path)
            data = test_dict[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
            targets = test_dict[b'labels']
            
            # Select first num_samples images
            if num_samples > len(data):
                st.warning(f"Requested {num_samples} samples, but only {len(data)} available. Using all available samples.")
                num_samples = len(data)
            indices = [i for i in range(num_samples)]
            sample_data = data[indices]
            sample_targets = [targets[i] for i in indices]
            
            return sample_data, sample_targets, labels, True
        else:
            st.warning(f"Dataset not found: {data_path}")
            return None, None, None, False
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None, None, False

def fgsm_attack(image, epsilon, data_grad):
    """Fast Gradient Sign Method attack"""
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(model, image, label, epsilon=8/255, alpha=2/255, num_iter=10):
    """Projected Gradient Descent attack"""
    criterion = nn.CrossEntropyLoss()
    
    # Start with a copy of the original image
    perturbed_image = image.clone().detach()
    
    for i in range(num_iter):
        perturbed_image.requires_grad_(True)
        
        # Forward pass - add batch dimension
        output, _ = model(perturbed_image.unsqueeze(0))
        loss = criterion(output, label)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update perturbation
        data_grad = perturbed_image.grad.data
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        
        # Project perturbation
        eta = torch.clamp(perturbed_image - image, -epsilon, epsilon)
        perturbed_image = torch.clamp(image + eta, 0, 1).detach()
    
    return perturbed_image

def predict_image(model, image_tensor, labels):
    """Make prediction on image"""
    with torch.no_grad():
        output, features = model(image_tensor.unsqueeze(0))
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities[0], 5)
        top5_predictions = [(labels[idx], prob.item()) for idx, prob in zip(top5_indices, top5_prob)]
        
        return prediction, confidence, top5_predictions, features

def visualize_perturbation(original, perturbed, epsilon):
    """Visualize the perturbation"""
    perturbation = perturbed - original
    
    # Normalize perturbation for visualization
    perturbation_vis = (perturbation + epsilon) / (2 * epsilon)
    perturbation_vis = torch.clamp(perturbation_vis, 0, 1)
    
    return perturbation_vis

def main():
    # Main header
    st.markdown('<h1 class="main-header">🛡️ CIFAR-10 Adversarial Training Demo</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    This demo showcases **Free Adversarial Training** on CIFAR-10 dataset using Wide ResNet architecture.
    Explore how models with different repetitions (Rep2 vs Rep8) perform against various adversarial attacks and compare robustness vs accuracy trade-offs.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection - includes both rep8 and rep2 models
    model_options = {
        "Best Model (Rep8)": "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/experiments_cifar10/wrn_32_10_eps8_step2_rep8/model_best.pth.tar",
        "Epoch 25 (Rep8)": "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/experiments_cifar10/wrn_32_10_eps8_step2_rep8/checkpoint_epoch25.pth.tar",
        "Epoch 21 (Rep8)": "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/experiments_cifar10/wrn_32_10_eps8_step2_rep8/checkpoint_epoch21.pth.tar",
        "Epoch 20 (Rep8)": "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/experiments_cifar10/wrn_32_10_eps8_step2_rep8/checkpoint_epoch20.pth.tar",
        "Best Model (Rep2)": "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/experiments_cifar10/wrn_32_10_eps8_step2_rep2/model_best.pth.tar",
        "Epoch 25 (Rep2)": "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/experiments_cifar10/wrn_32_10_eps8_step2_rep2/checkpoint_epoch25.pth.tar",
        "Epoch 20 (Rep2)": "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/experiments_cifar10/wrn_32_10_eps8_step2_rep2/checkpoint_epoch20.pth.tar",
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model Checkpoint",
        list(model_options.keys()),
        help="Choose which trained model to use (Rep8 vs Rep2 comparison)"
    )
    
    model_path = model_options[selected_model]
    
    # Display model info
    st.sidebar.markdown("### Model Information")
    rep_count = "8" if "Rep8" in selected_model else "2"
    st.sidebar.info(
        "**Architecture**: Wide ResNet-32-10\n\n"
        "**Training**: Free Adversarial Training\n\n"
        f"**Repetitions**: {rep_count}\n\n"
        "**Perturbation Budget**: ε = 8/255\n\n"
        "**Dataset**: CIFAR-10 (10 classes)"
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model, model_loaded = load_model(model_path)
    
    if not model_loaded:
        st.error("Failed to load model. Please check the model path.")
        return
    
    # Load sample images
    with st.spinner("Loading CIFAR-10 sample images..."):
        sample_data, sample_targets, labels, data_loaded = load_sample_images(100)
    
    if not data_loaded:
        st.error("Failed to load CIFAR-10 dataset.")
        return
    
    st.success(f"✅ Model and CIFAR-10 dataset loaded successfully!")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Classification", "⚔️ Adversarial Attacks", "📊 Model Analysis", "📚 About"])
    
    with tab1:
        st.header("Image Classification")
        st.markdown("Select an image to see how the model classifies it.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Image selection
            image_idx = st.selectbox(
                "Select Image",
                range(len(sample_data)),
                format_func=lambda x: f"Image {x+1} - {labels[sample_targets[x]]}"
            )
            
            # Display selected image
            image = sample_data[image_idx]
            true_label = sample_targets[image_idx]
            
            st.image(image, caption=f"True Label: {labels[true_label]}", width=200)
            
        with col2:
            # Convert to tensor for prediction
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
            
            image_tensor = transform(image)
            
            # Make prediction
            pred_idx, confidence, top5_preds, features = predict_image(model, image_tensor, labels)
            
            # Display results
            st.subheader("Prediction Results")
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Predicted Class", labels[pred_idx])
                st.metric("Confidence", f"{confidence}")
                
            with col2b:
                is_correct = pred_idx == true_label
                st.metric("Accuracy", "✅ Correct" if is_correct else "❌ Incorrect")
                st.metric("True Class", labels[true_label])
            
            # Top 5 predictions
            st.subheader("Top 5 Predictions")
            for i, (label, prob) in enumerate(top5_preds):
                st.write(f"{i+1}. **{label}**: {prob}")
    
    with tab2:
        st.header("Adversarial Attacks")
        st.markdown("Test the model's robustness against adversarial perturbations.")
        
        # Attack configuration
        st.subheader("Attack Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            attack_type = st.selectbox(
                "Attack Type",
                ["FGSM", "PGD"],
                help="Fast Gradient Sign Method or Projected Gradient Descent"
            )
            
        with col2:
            epsilon = st.slider(
                "Epsilon (perturbation strength)",
                min_value=0.0,
                max_value=0.1,
                value=0.031,  # 8/255
                step=0.001,
                help="Maximum perturbation magnitude"
            )
            
        with col3:
            if attack_type == "PGD":
                num_iter = st.slider(
                    "PGD Iterations",
                    min_value=1,
                    max_value=20,
                    value=20,
                    help="Number of PGD iterations"
                )
        
        # Image selection for attack
        attack_image_idx = st.selectbox(
            "Select Image for Attack",
            range(len(sample_data)),
            format_func=lambda x: f"Image {x+1} - {labels[sample_targets[x]]}",
            key="attack_image"
        )
        
        if st.button("🚀 Launch Attack", type="primary"):
            with st.spinner("Generating adversarial example..."):
                # Prepare image
                attack_image = sample_data[attack_image_idx]
                true_label_idx = sample_targets[attack_image_idx]
                
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor()
                ])
                
                image_tensor = transform(attack_image)
                label_tensor = torch.tensor([true_label_idx])
                
                # Generate adversarial example
                if attack_type == "FGSM":
                    # FGSM attack
                    image_tensor.requires_grad_(True)
                    output, _ = model(image_tensor.unsqueeze(0))
                    loss = F.cross_entropy(output, label_tensor)
                    
                    model.zero_grad()
                    loss.backward()
                    data_grad = image_tensor.grad.data
                    
                    perturbed_image = fgsm_attack(image_tensor, epsilon, data_grad)
                else:
                    # PGD attack
                    perturbed_image = pgd_attack(model, image_tensor, label_tensor, epsilon, num_iter=num_iter)
                
                # Make predictions
                orig_pred, orig_conf, orig_top5, _ = predict_image(model, image_tensor, labels)
                adv_pred, adv_conf, adv_top5, _ = predict_image(model, perturbed_image, labels)
                
                # Visualize results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Original Image")
                    # orig_img_np = image_tensor.detach().permute(1, 2, 0).numpy()
                    orig_img_np = image_tensor.clone().detach().permute(1, 2, 0).numpy()
                    st.image(orig_img_np, caption=f"Pred: {labels[orig_pred]} ({orig_conf})")
                    
                    st.markdown('<div class="defense-info">', unsafe_allow_html=True)
                    st.write(f"**True Label**: {labels[true_label_idx]}")
                    st.write(f"**Predicted**: {labels[orig_pred]}")
                    st.write(f"**Confidence**: {orig_conf}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.subheader("Adversarial Image")
                    # adv_img_np = perturbed_image.permute(1, 2, 0).numpy()
                    adv_img_np = perturbed_image.clone().detach().permute(1, 2, 0).numpy()
                    st.image(adv_img_np, caption=f"Pred: {labels[adv_pred]} ({adv_conf})")
                    
                    st.markdown('<div class="attack-info">', unsafe_allow_html=True)
                    st.write(f"**True Label**: {labels[true_label_idx]}")
                    st.write(f"**Predicted**: {labels[adv_pred]}")
                    st.write(f"**Confidence**: {adv_conf}")
                    attack_success = orig_pred != adv_pred
                    st.write(f"**Attack Success**: {'✅ Yes' if attack_success else '❌ No'}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.subheader("Perturbation")
                    perturbation_vis = visualize_perturbation(image_tensor, perturbed_image, epsilon)
                    # pert_img_np = perturbation_vis.permute(1, 2, 0).numpy()
                    pert_img_np = perturbation_vis.clone().detach().permute(1, 2, 0).numpy()
                    st.image(pert_img_np, caption=f"Perturbation (ε={epsilon:.3f})")
                    
                    # Perturbation statistics
                    perturbation = perturbed_image - image_tensor
                    l2_norm = torch.norm(perturbation).item()
                    linf_norm = torch.max(torch.abs(perturbation)).item()
                    
                    st.write(f"**L2 Norm**: {l2_norm:.4f}")
                    st.write(f"**L∞ Norm**: {linf_norm:.4f}")
                    st.write(f"**Max Epsilon**: {epsilon:.3f}")
        
        # Add Rep2 vs Rep8 Model Comparison Section
        st.markdown("---")
        st.subheader("🔄 Rep2 vs Rep8 Model Comparison")
        st.markdown("Compare the robustness of models trained with different repetitions on the same image.")
        
        # Model comparison configuration
        col1, col2 = st.columns(2)
        
        with col1:
            comp_attack_type = st.selectbox(
                "Comparison Attack Type",
                ["FGSM", "PGD"],
                help="Attack type for model comparison",
                key="comp_attack"
            )
            
        with col2:
            comp_epsilon = st.slider(
                "Comparison Epsilon",
                min_value=0.0,
                max_value=0.1,
                value=0.031,
                step=0.001,
                help="Perturbation strength for comparison",
                key="comp_epsilon"
            )
        
        # Image selection for comparison
        comp_image_idx = st.selectbox(
            "Select Image for Model Comparison",
            range(len(sample_data)),
            format_func=lambda x: f"Image {x+1} - {labels[sample_targets[x]]}",
            key="comp_image"
        )
        
        if st.button("🔄 Compare Models", type="secondary"):
            with st.spinner("Comparing Rep2 and Rep8 models..."):
                # Load Rep2 and Rep8 models
                rep2_model_path = "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/experiments_cifar10/wrn_32_10_eps8_step2_rep2/model_best.pth.tar"
                rep8_model_path = "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/experiments_cifar10/wrn_32_10_eps8_step2_rep8/model_best.pth.tar"
                
                rep2_model, rep2_loaded = load_model(rep2_model_path)
                rep8_model, rep8_loaded = load_model(rep8_model_path)
                
                if rep2_loaded and rep8_loaded:
                    # Prepare image
                    comp_image = sample_data[comp_image_idx]
                    comp_true_label_idx = sample_targets[comp_image_idx]
                    
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.ToTensor()
                    ])
                    
                    comp_image_tensor = transform(comp_image)
                    comp_label_tensor = torch.tensor([comp_true_label_idx])
                    
                    # Generate adversarial examples for both models
                    results = {}
                    for model_name, model in [("Rep2", rep2_model), ("Rep8", rep8_model)]:
                        if comp_attack_type == "FGSM":
                            # FGSM attack
                            temp_image = comp_image_tensor.clone().detach().requires_grad_(True)
                            output, _ = model(temp_image.unsqueeze(0))
                            loss = F.cross_entropy(output, comp_label_tensor)
                            
                            model.zero_grad()
                            loss.backward()
                            data_grad = temp_image.grad.data
                            
                            perturbed_image = fgsm_attack(temp_image, comp_epsilon, data_grad)
                        else:
                            # PGD attack
                            perturbed_image = pgd_attack(model, comp_image_tensor, comp_label_tensor, comp_epsilon, num_iter=20)
                        
                        # Make predictions
                        orig_pred, orig_conf, _, _ = predict_image(model, comp_image_tensor, labels)
                        adv_pred, adv_conf, _, _ = predict_image(model, perturbed_image, labels)
                        
                        results[model_name] = {
                            'perturbed_image': perturbed_image,
                            'orig_pred': orig_pred,
                            'orig_conf': orig_conf,
                            'adv_pred': adv_pred,
                            'adv_conf': adv_conf,
                            'attack_success': orig_pred != adv_pred
                        }
                    
                    # Display comparison results
                    st.subheader("Model Comparison Results")
                    
                    # Original image
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Original Image**")
                        orig_img_np = comp_image_tensor.clone().detach().permute(1, 2, 0).numpy()
                        st.image(orig_img_np, caption=f"True: {labels[comp_true_label_idx]}")
                        
                        st.markdown('<div class="defense-info">', unsafe_allow_html=True)
                        st.write(f"**Rep2 Pred**: {labels[results['Rep2']['orig_pred']]} ({results['Rep2']['orig_conf']})")
                        st.write(f"**Rep8 Pred**: {labels[results['Rep8']['orig_pred']]} ({results['Rep8']['orig_conf']})")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Rep2 Adversarial**")
                        rep2_adv_img = results['Rep2']['perturbed_image'].clone().detach().permute(1, 2, 0).numpy()
                        st.image(rep2_adv_img, caption=f"Pred: {labels[results['Rep2']['adv_pred']]} ({results['Rep2']['adv_conf']})")
                        
                        st.markdown('<div class="attack-info">', unsafe_allow_html=True)
                        st.write(f"**Attack Success**: {'✅ Yes' if results['Rep2']['attack_success'] else '❌ No'}")
                        st.write(f"**Confidence Drop**: {results['Rep2']['orig_conf'] - results['Rep2']['adv_conf']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown("**Rep8 Adversarial**")
                        rep8_adv_img = results['Rep8']['perturbed_image'].clone().detach().permute(1, 2, 0).numpy()
                        st.image(rep8_adv_img, caption=f"Pred: {labels[results['Rep8']['adv_pred']]} ({results['Rep8']['adv_conf']})")
                        
                        st.markdown('<div class="attack-info">', unsafe_allow_html=True)
                        st.write(f"**Attack Success**: {'✅ Yes' if results['Rep8']['attack_success'] else '❌ No'}")
                        st.write(f"**Confidence Drop**: {results['Rep8']['orig_conf'] - results['Rep8']['adv_conf']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Robustness comparison summary
                    st.subheader("📊 Robustness Comparison Summary")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Rep2 Model", 
                                 f"{'Robust' if not results['Rep2']['attack_success'] else 'Vulnerable'}", 
                                 f"{results['Rep2']['adv_conf']} confidence")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.metric("Rep8 Model", 
                                 f"{'Robust' if not results['Rep8']['attack_success'] else 'Vulnerable'}", 
                                 f"{results['Rep8']['adv_conf']} confidence")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Winner determination
                    rep2_score = (1 - int(results['Rep2']['attack_success'])) + results['Rep2']['adv_conf']
                    rep8_score = (1 - int(results['Rep8']['attack_success'])) + results['Rep8']['adv_conf']
                    print(rep2_score, rep8_score)
                    if results['Rep2']['attack_success'] and results['Rep8']['attack_success']:
                        winner = "Both models were robust against the attack!"
                        winner_color = "✅"
                    if not results['Rep2']['attack_success'] and not results['Rep8']['attack_success']:
                        winner = "Both models were vulnerable to the attack!"
                        winner_color = "❌"
                    elif rep2_score > rep8_score:
                        winner = "Rep2 model shows better robustness for this example!"
                        winner_color = "🥇"
                    elif rep8_score > rep2_score:
                        winner = "Rep8 model shows better robustness for this example!"
                        winner_color = "🥇"
                    else:
                        winner = "Both models show similar robustness for this example!"
                        winner_color = "🤝"
                    
                    st.info(f"{winner_color} {winner}")
                    
                else:
                    st.error("Failed to load one or both models for comparison.")
    
    with tab3:
        st.header("Model Analysis")
        st.markdown("Analyze model performance and robustness metrics.")
        
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Model Architecture", "Wide ResNet-32-10")
            st.metric("Training Method", "Free Adversarial Training")
            current_rep = "8" if "Rep8" in selected_model else "2"
            st.metric("Current Model Repetitions", current_rep)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Dataset", "CIFAR-10")
            st.metric("Number of Classes", "10")
            st.metric("Input Resolution", "32×32")
            st.metric("Parameters", "~36M")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Robustness evaluation
        st.subheader("🛡️ Robustness Evaluation")
        
        if st.button("Run Robustness Test", type="primary"):
            with st.spinner("Evaluating model robustness..."):
                # Test on a subset of images
                test_indices = np.random.choice(len(sample_data), min(20, len(sample_data)), replace=False)
                
                epsilons = [0.0, 0.008, 0.016, 0.024, 0.031, 0.047, 0.063]
                accuracies = []
                
                for eps in epsilons:
                    correct = 0
                    total = 0
                    
                    for idx in test_indices:
                        image = sample_data[idx]
                        true_label = sample_targets[idx]
                        
                        transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor()
                        ])
                        
                        image_tensor = transform(image)
                        
                        if eps == 0.0:
                            # Clean accuracy
                            pred_idx, _, _, _ = predict_image(model, image_tensor, labels)
                        else:
                            # Adversarial accuracy
                            label_tensor = torch.tensor([true_label])
                            perturbed_image = pgd_attack(model, image_tensor, label_tensor, eps, num_iter=20)
                            pred_idx, _, _, _ = predict_image(model, perturbed_image, labels)
                        
                        if pred_idx == true_label:
                            correct += 1
                        total += 1
                    
                    accuracy = correct / total
                    accuracies.append(accuracy)
                
                # Plot robustness curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epsilons,
                    y=accuracies,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Model Robustness vs Perturbation Strength",
                    xaxis_title="Epsilon (L∞ perturbation)",
                    yaxis_title="Accuracy",
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display accuracy table
                accuracy_df = {
                    "Epsilon": epsilons,
                    "Accuracy": [f"{acc}" for acc in accuracies]
                }
                st.table(accuracy_df)
        
        # Add Rep2 vs Rep8 Bulk Comparison
        st.markdown("---")
        st.subheader("🔄 Rep2 vs Rep8 Bulk Robustness Comparison")
        st.markdown("Compare the overall robustness of Rep2 vs Rep8 models across multiple images and epsilon values.")
        
        if st.button("🚀 Run Bulk Comparison", type="secondary"):
            with st.spinner("Running comprehensive comparison between Rep2 and Rep8 models..."):
                # Load both models
                rep2_model_path = "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/experiments_cifar10/wrn_32_10_eps8_step2_rep2/model_best.pth.tar"
                rep8_model_path = "/mlcv2/WorkingSpace/Personal/tuongbck/cs410/experiments_cifar10/wrn_32_10_eps8_step2_rep8/model_best.pth.tar"
                
                rep2_model, rep2_loaded = load_model(rep2_model_path)
                rep8_model, rep8_loaded = load_model(rep8_model_path)
                
                if rep2_loaded and rep8_loaded:
                    # Test parameters
                    test_indices = np.random.choice(len(sample_data), len(sample_data), replace=False)
                    epsilons = [0.0, 0.016, 0.031, 0.047, 0.063]
                    
                    rep2_accuracies = []
                    rep8_accuracies = []
                    
                    for eps in epsilons:
                        rep2_correct = 0
                        rep8_correct = 0
                        total = len(test_indices)
                        
                        for idx in tqdm(test_indices):
                            image = sample_data[idx]
                            true_label = sample_targets[idx]
                            
                            transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.ToTensor()
                            ])
                            
                            image_tensor = transform(image)
                            label_tensor = torch.tensor([true_label])
                            
                            # Test Rep2 model
                            if eps == 0.0:
                                pred_idx, _, _, _ = predict_image(rep2_model, image_tensor, labels)
                            else:
                                perturbed_image = pgd_attack(rep2_model, image_tensor, label_tensor, eps, num_iter=20)
                                pred_idx, _, _, _ = predict_image(rep2_model, perturbed_image, labels)
                            
                            if pred_idx == true_label:
                                rep2_correct += 1
                            
                            # Test Rep8 model
                            if eps == 0.0:
                                pred_idx, _, _, _ = predict_image(rep8_model, image_tensor, labels)
                            else:
                                perturbed_image = pgd_attack(rep8_model, image_tensor, label_tensor, eps, num_iter=20)
                                pred_idx, _, _, _ = predict_image(rep8_model, perturbed_image, labels)
                            
                            if pred_idx == true_label:
                                rep8_correct += 1
                        print(rep2_correct, rep8_correct, total)
                        rep2_accuracies.append(rep2_correct / total)
                        rep8_accuracies.append(rep8_correct / total)
                    
                    # Plot comparison
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=epsilons,
                        y=rep2_accuracies,
                        mode='lines+markers',
                        name='Rep2 Model',
                        line=dict(color='#ff7f0e', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=epsilons,
                        y=rep8_accuracies,
                        mode='lines+markers',
                        name='Rep8 Model',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title="Rep2 vs Rep8 Model Robustness Comparison",
                        xaxis_title="Epsilon (L∞ perturbation)",
                        yaxis_title="Accuracy",
                        yaxis=dict(range=[0, 1]),
                        height=500,
                        legend=dict(x=0.7, y=0.95)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Comparison summary table
                    comparison_data = {
                        "Epsilon": epsilons,
                        "Rep2 Accuracy": [f"{acc}" for acc in rep2_accuracies],
                        "Rep8 Accuracy": [f"{acc}" for acc in rep8_accuracies],
                        "Difference": [f"{rep8_acc - rep2_acc:+.2%}" for rep2_acc, rep8_acc in zip(rep2_accuracies, rep8_accuracies)]
                    }
                    
                    st.markdown("### 📊 Detailed Comparison Results")
                    st.dataframe(comparison_data, use_container_width=True)
                    
                    # Overall winner
                    rep2_avg = np.mean(rep2_accuracies[1:])  # Exclude clean accuracy
                    rep8_avg = np.mean(rep8_accuracies[1:])  # Exclude clean accuracy
                    
                    if rep8_avg > rep2_avg:
                        st.success(f"🏆 **Rep8 model wins overall!** Average adversarial accuracy: {rep8_avg} vs {rep2_avg}")
                    elif rep2_avg > rep8_avg:
                        st.success(f"🏆 **Rep2 model wins overall!** Average adversarial accuracy: {rep2_avg} vs {rep8_avg}")
                    else:
                        st.info(f"🤝 **Both models perform similarly!** Rep2: {rep2_avg}, Rep8: {rep8_avg}")
                
                else:
                    st.error("Failed to load one or both models for bulk comparison.")
    
    with tab4:
        st.header("About This Demo")
        
        st.markdown("""
        ## 🔬 Free Adversarial Training: Rep2 vs Rep8 Comparison
        
        This demo showcases **Free Adversarial Training** implementations for CIFAR-10 dataset, comparing models trained with **2 repetitions** vs **8 repetitions**. 
        The approach trains neural networks to be robust against adversarial attacks while maintaining computational efficiency.
        
        ### 🏗️ Architecture
        - **Model**: Wide ResNet-32-10 (Wide Residual Network with 32 layers and width factor 10)
        - **Parameters**: Approximately 36 million parameters
        - **Input**: 32×32 RGB images
        - **Output**: Class probabilities for CIFAR-10 (10 classes)
        
        ### 🛡️ Adversarial Training Variants
        
        #### Rep2 Model
        - Uses **2 repetitions** per minibatch during training
        - Faster training with moderate robustness improvement
        - Good balance between training efficiency and adversarial robustness
        
        #### Rep8 Model  
        - Uses **8 repetitions** per minibatch during training
        - Enhanced robustness at the cost of longer training time
        - Stronger defense against gradient-based attacks
        
        Both models:
        - Generate adversarial examples during training using FGSM (Fast Gradient Sign Method)
        - Train on both clean and adversarial examples
        - Use perturbation budget ε = 8/255
        
        ### ⚔️ Attack Methods
        1. **FGSM (Fast Gradient Sign Method)**:
           - Single-step attack
           - Fast but less sophisticated
           - Uses sign of gradient for perturbation direction
        
        2. **PGD (Projected Gradient Descent)**:
           - Multi-step iterative attack
           - More sophisticated and effective
           - Projects perturbations to maintain L∞ constraint
        
        ### 📊 Key Features
        - **Real-time inference** on CIFAR-10 images
        - **Interactive adversarial attack generation**
        - **Robustness evaluation** across different perturbation strengths
        - **Visualization** of perturbations and model predictions
        
        ### 🔗 Technical Details
        - **Training Method**: Free Adversarial Training with n_repeats=8
        - **Perturbation Budget**: ε = 8/255 (L∞ norm)
        - **Optimizer**: SGD with momentum=0.9, weight_decay=0.0002
        - **Learning Rate Schedule**: Step decay every 30 epochs
        - **Dataset**: CIFAR-10 (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        
        ### 📚 References
        - Shafahi et al. "Adversarial Training for Free!" NeurIPS 2019
        - Zagoruyko & Komodakis "Wide Residual Networks" BMVC 2016
        - Goodfellow et al. "Explaining and Harnessing Adversarial Examples" ICLR 2015
        """)
        
        st.markdown("---")
        st.markdown("*Built with Streamlit, PyTorch, and ❤️ - Focused on CIFAR-10 Rep8 Model*")

if __name__ == "__main__":
    main()
