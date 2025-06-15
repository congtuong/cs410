#!/bin/bash

# Simple script to run the CIFAR-10 Adversarial Training Demo
# Make sure you have installed the required dependencies:
# pip install streamlit torch torchvision numpy matplotlib seaborn plotly Pillow

echo "Starting CIFAR-10 Adversarial Training Demo..."
echo "Model: Wide ResNet-32-10 with 8 repetitions"
echo "Dataset: CIFAR-10"
echo ""
echo "Open your browser and navigate to the URL shown below:"
echo ""

streamlit run streamlit_demo.py --server.port 8501 --server.address 0.0.0.0
