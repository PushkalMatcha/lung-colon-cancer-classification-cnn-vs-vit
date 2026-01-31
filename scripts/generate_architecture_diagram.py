"""
Generate high-resolution ER/architecture diagram for the lung-colon classification project.
Outputs a 300 DPI PNG suitable for research papers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Set up high-resolution figure (300 DPI for publication quality)
fig, ax = plt.subplots(figsize=(18, 14), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

# Define color scheme
color_data = '#E8F4F8'      # Light blue for data
color_process = '#FFF4E6'   # Light orange for processing
color_model = '#E8F5E9'     # Light green for models
color_output = '#F3E5F5'    # Light purple for outputs
edge_color = '#424242'      # Dark gray for edges

def draw_box(ax, x, y, width, height, label, color, fontsize=11, fontweight='bold'):
    """Draw a rounded rectangle box with label."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.15",
        edgecolor=edge_color,
        facecolor=color,
        linewidth=2.5
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, label,
            ha='center', va='center', fontsize=fontsize, fontweight=fontweight, wrap=True)
    return box

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=edge_color,
        linewidth=2.5,
        mutation_scale=25
    )
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9, linewidth=1))

# Title
ax.text(5, 15.3, 'Lung & Colon Cancer Classification Pipeline', 
        ha='center', fontsize=18, fontweight='bold')
ax.text(5, 14.8, 'Deep Learning Architecture & Workflow', 
        ha='center', fontsize=13, style='italic')

# Layer 1: Dataset
draw_box(ax, 3, 13.5, 4, 1, 'LC25000 Dataset\n25,000 histopathological images\n5 classes (colon_aca, colon_n, lung_aca, lung_n, lung_scc)', 
         color_data, fontsize=10)

# Layer 2: Data preprocessing
draw_arrow(ax, 5, 13.5, 5, 12.5, '')
draw_box(ax, 1.5, 11.5, 2.8, 0.8, 'Data Split\n70% Train\n15% Val\n15% Test', color_process, fontsize=10)
draw_box(ax, 5.7, 11.5, 2.8, 0.8, 'Optional Stain\nNormalization\n(staintools)', color_process, fontsize=10)

# Layer 3: Data loaders
draw_arrow(ax, 2.9, 11.5, 2.1, 10.4, '')
draw_arrow(ax, 4.3, 11.5, 4.9, 10.4, '')
draw_arrow(ax, 7.1, 11.5, 7.7, 10.4, 'if enabled')
draw_box(ax, 1, 9.6, 2.2, 0.8, 'Train Loader\n17,500 images', color_process, fontsize=10)
draw_box(ax, 3.8, 9.6, 2.2, 0.8, 'Val Loader\n3,750 images', color_process, fontsize=10)
draw_box(ax, 6.6, 9.6, 2.2, 0.8, 'Test Loader\n3,750 images', color_process, fontsize=10)

# Layer 4: Training process
draw_arrow(ax, 2.1, 9.6, 3.2, 8.6, '')
draw_arrow(ax, 4.9, 9.6, 4.5, 8.6, '')
draw_box(ax, 2.5, 7.8, 3.5, 0.8, 'Training Loop\n(scripts/train.py)\nAdam Optimizer | CrossEntropyLoss', 
         color_process, fontsize=10)

# Layer 5: Model architectures (side by side)
draw_arrow(ax, 3.5, 7.8, 2.2, 6.7, '')
draw_arrow(ax, 5, 7.8, 7.3, 6.7, '')

draw_box(ax, 0.3, 5.5, 3.8, 1.2, 
         'ResNet50\n(torchvision.models)\nPretrained on ImageNet\nCustom FC: 2048 → 5 classes',
         color_model, fontsize=10)
draw_box(ax, 5.3, 5.5, 4.1, 1.2,
         'Vision Transformer (ViT)\nvit_b_16 (torchvision)\nPretrained on ImageNet\nCustom Head: 768 → 5 classes',
         color_model, fontsize=10)

# Four model variants
draw_arrow(ax, 1.4, 5.5, 1.45, 4.3, '')
draw_arrow(ax, 2.9, 5.5, 3.6, 4.3, '')
draw_arrow(ax, 6.5, 5.5, 6, 4.3, '')
draw_arrow(ax, 8.2, 5.5, 8.2, 4.3, '')

draw_box(ax, 0.5, 3.5, 1.9, 0.8, 'resnet\nAcc: 97.17%', color_model, fontsize=10)
draw_box(ax, 2.7, 3.5, 1.9, 0.8, 'resnet_stain_norm\nAcc: 97.17%', color_model, fontsize=10)
draw_box(ax, 5.1, 3.5, 1.9, 0.8, 'vit\nAcc: 97.36%', color_model, fontsize=10)
draw_box(ax, 7.3, 3.5, 1.9, 0.8, 'vit_stain_norm\nAcc: 97.36%', color_model, fontsize=10)

# Layer 6: Checkpoints
draw_arrow(ax, 1.45, 3.5, 2.5, 2.8, '')
draw_arrow(ax, 3.6, 3.5, 4, 2.8, '')
draw_arrow(ax, 6, 3.5, 5.5, 2.8, '')
draw_arrow(ax, 8.2, 3.5, 6.5, 2.8, '')

draw_box(ax, 1, 2.2, 6.5, 0.6, 
         'Saved Checkpoints (results/saved_models/best_model_*.pth)\nPyTorch state_dict with model weights',
         color_output, fontsize=10)

# Layer 7: Evaluation
draw_arrow(ax, 4.25, 2.2, 4.25, 1.4, '')
draw_arrow(ax, 7.7, 9.6, 8.8, 1.4, 'Test Data')
draw_box(ax, 2.75, 0.8, 3, 0.6, 'Model Evaluation\n(scripts/eval.py)', color_process, fontsize=11)

# Layer 8: Results and outputs
draw_arrow(ax, 3.5, 0.8, 2.1, -0.3, '')
draw_arrow(ax, 4.25, 0.8, 5.6, -0.3, '')
draw_arrow(ax, 5.2, 0.8, 8.6, -0.3, '')

draw_box(ax, 0.2, -1, 3.9, 0.7,
         'Combined JSON\n(all_models_history.json)\nAccuracy, Classification Report,\nConfusion Matrix',
         color_output, fontsize=9)
draw_box(ax, 4.4, -1, 2.8, 0.7,
         'Confusion Matrix\nPlots (PNG)\nPer-model heatmaps',
         color_output, fontsize=9)
draw_box(ax, 7.5, -1, 2.3, 0.7,
         'Training Curves\nPlots (PNG)\nLoss & Accuracy',
         color_output, fontsize=9)

# Add legend
legend_y = -2
ax.text(0.5, legend_y, 'Legend:', fontsize=11, fontweight='bold')
draw_box(ax, 0.5, legend_y - 0.6, 1.4, 0.4, 'Data', color_data, fontsize=10, fontweight='normal')
draw_box(ax, 2.2, legend_y - 0.6, 1.4, 0.4, 'Process', color_process, fontsize=10, fontweight='normal')
draw_box(ax, 3.9, legend_y - 0.6, 1.4, 0.4, 'Model', color_model, fontsize=10, fontweight='normal')
draw_box(ax, 5.6, legend_y - 0.6, 1.4, 0.4, 'Output', color_output, fontsize=10, fontweight='normal')

# Add project metadata
ax.text(9.5, legend_y - 0.8, 'Project: Lung & Colon Cancer Classification', 
        fontsize=9, style='italic', ha='right')
ax.text(9.5, legend_y - 1.2, 'Framework: PyTorch + torchvision', 
        fontsize=9, style='italic', ha='right')
ax.text(9.5, legend_y - 1.6, 'Dataset: LC25000 (25k images, 5 classes)', 
        fontsize=9, style='italic', ha='right')

plt.tight_layout()

# Save the diagram
output_dir = 'results/plots'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'architecture_diagram_300dpi.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Architecture diagram saved to: {output_path}")
print(f"  Resolution: 300 DPI")
print(f"  Format: PNG")

# Also save a PDF version for vector graphics (better for papers)
output_path_pdf = os.path.join(output_dir, 'architecture_diagram.pdf')
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Vector version saved to: {output_path_pdf}")
print(f"  Format: PDF (scalable vector graphics)")

plt.close()
