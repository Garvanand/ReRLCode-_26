import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# --- Setup canvas ---
fig, ax = plt.subplots(figsize=(22, 10))
fig.patch.set_facecolor('#F8F9FA')
ax.set_xlim(0, 22)
ax.set_ylim(0, 10)
ax.axis('off')

# --- Helper functions ---
def draw_box(x, y, w, h, text, fc, ec, fontsize=11, lw=1.8):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=fontsize, family='DejaVu Sans')

def draw_arrow(x1, y1, x2, y2, text="", color='black', style='-', rad=0.0):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->',
                            mutation_scale=14,
                            linewidth=1.5,
                            linestyle=style,
                            color=color,
                            connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(arrow)
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.2, text,
                fontsize=9, ha='center', family='DejaVu Sans')

# --- COLUMN 1: Inputs ---
draw_box(0.5, 7.0, 1.5, 1.2, "Image\nI", "#DBEAFE", "#1D4ED8")
draw_box(0.5, 4.5, 1.5, 1.2, "Caption\nC⁺", "#EDE9FE", "#5B21B6")
draw_box(0.4, 2.2, 1.7, 1.8, "SNCG\nNeg. Caption\nGenerator", "#FEF3C7", "#B45309")
draw_box(0.5, 0.3, 1.5, 1.2, "Caption\nC⁻", "#FDE68A", "#B45309")

draw_arrow(1.25, 4.5, 1.25, 3.9, "generates", style='--')
draw_arrow(1.25, 2.2, 1.25, 1.5)

# --- COLUMN 2: Encoders ---
draw_box(3.0, 6.0, 2.0, 2.5,
         "Vision Encoder\n(ViT-B/16)\nPre-trained\nOutput: V ∈ ℝ^(Nv+1)×d",
         "#DBEAFE", "#1D4ED8")

draw_box(3.0, 2.5, 2.0, 2.5,
         "Text Encoder\n(BERT-base)\nPre-trained\nOutputs: T⁺, T⁻ ∈ ℝ^(Nt+1)×d",
         "#EDE9FE", "#5B21B6")

draw_arrow(2.0, 7.6, 3.0, 7.2, "I")
draw_arrow(2.0, 5.1, 3.0, 4.8, "C⁺")
draw_arrow(2.0, 0.9, 3.0, 3.0, "C⁻", color='orange', style='--')

# --- COLUMN 3: STM ---
draw_box(5.5, 1.0, 8.0, 8.0, "Semantic Tension Module (STM)",
         "#FAF5FF", "#7C3AED", fontsize=13)

# Inner Boxes
draw_box(6.0, 6.5, 3.5, 1.8,
         "Cross-Attention (+)\nV̂⁺ = Attn(Q=V,K=T⁺,V=T⁺)\nT̂⁺ = Attn(Q=T⁺,K=V,V=V)\n4 heads, d_h=64",
         "#EDE9FE", "#7C3AED", fontsize=9)

draw_box(10.0, 6.5, 3.5, 1.8,
         "Cross-Attention (−)\nV̂⁻ = Attn(Q=V,K=T⁻,V=T⁻)\nT̂⁻ = Attn(Q=T⁻,K=V,V=V)\n4 heads, d_h=64",
         "#FEF3C7", "#B45309", fontsize=9)

draw_box(7.5, 4.2, 4.0, 1.5,
         "Discrepancy Δ\nΔV = V̂⁺ − V̂⁻\nΔT = T̂⁺ − T̂⁻\nd = MeanPool(ΔV) ∥ MeanPool(ΔT)",
         "#D1FAE5", "#065F46", fontsize=9)

draw_box(7.5, 2.3, 4.0, 1.5,
         "Tension Scoring Head\nτ = σ(wᵀϕ(d))\nτ≈1 inconsistent\nτ≈0 consistent",
         "#DCFCE7", "#166534", fontsize=9)

draw_box(11.8, 2.3, 1.5, 2.5,
         "Embedding\nRefinement\nṽ = v_cls + α·Pool(V̂⁺)\nt̃⁺ = t⁺_cls + α·Pool(T̂⁺)\n→ z_v, z_t⁺",
         "#FCE7F3", "#9D174D", fontsize=8)

# Arrows inside STM
draw_arrow(7.8, 6.5, 8.5, 5.7)
draw_arrow(11.8, 6.5, 10.5, 5.7)
draw_arrow(9.5, 5.7, 9.5, 4.9)
draw_arrow(9.5, 4.2, 9.5, 3.8)

# Encoder → STM
draw_arrow(5.0, 7.2, 6.0, 7.2, "V")
draw_arrow(5.0, 4.8, 6.0, 7.2, "T⁺")
draw_arrow(5.0, 3.0, 10.0, 7.2, "T⁻", style='--', color='orange')

# --- COLUMN 4: DACL Loss ---
draw_box(14.0, 2.0, 7.0, 6.5,
         "DACL Loss\n\nℒ_align (InfoNCE, τ_c=0.07)\nStandard contrastive loss\n\n"
         "+ λ₁·ℒ_band (λ₁=1.0)\nTension band [0.05,0.15]\n\n"
         "+ λ₂·ℒ_disc (λ₂=0.5)\nBCE on τ\n\n"
         "──────────────\nℒ_total = ℒ_align + λ₁ℒ_band + λ₂ℒ_disc",
         "#F0FDF4", "#166534", fontsize=10)

# Arrows into loss
draw_arrow(13.3, 3.5, 14.0, 5.5, "z_v,z_t⁺", color='#9D174D')
draw_arrow(13.3, 3.0, 14.0, 4.5, "τ", style='--', color='green')

# --- Title ---
plt.title("Fig. 1 – High-Level Architecture of DisAlign",
          fontsize=16, family='DejaVu Sans')

# --- Legend ---
legend_patches = [
    mpatches.Patch(color="#DBEAFE", label="Vision stream"),
    mpatches.Patch(color="#EDE9FE", label="Language (+)"),
    mpatches.Patch(color="#FEF3C7", label="Negative (−)"),
    mpatches.Patch(color="#FAF5FF", label="STM module"),
    mpatches.Patch(color="#F0FDF4", label="DACL Loss"),
]

ax.legend(handles=legend_patches,
          loc='lower center',
          bbox_to_anchor=(0.5, -0.05),
          ncol=5,
          frameon=False)

# --- Save ---
plt.savefig("fig1_disalign_architecture.png", dpi=200, bbox_inches='tight')
plt.close()