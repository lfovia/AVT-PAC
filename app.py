import gradio as gr
import pandas as pd
from inference import run_streaming_inference, load_models
import os
import base64

# Default actions for the demo
DEFAULT_ACTIONS = [
    "walking", "running", "opening door", "drinking", "talking",
    "sitting", "standing", "climbing stairs", "falling", "kicking",
    "punching", "waving", "clapping", "reading", "writing",
    "eating", "cooking", "playing guitar", "playing piano", "dancing",
    "jumping", "bending", "catching", "throwing", "pushing"
]

# Paper information
PAPER_INFO = {
    "title": "Audio-Visual Activity Recognition with TCN-GRU Architecture",
    "authors": "Your Name, Co-author Name, Another Author",
    "journal": "arXiv preprint arXiv:2026.12345",
    "year": "2026",
    "abstract": """
    We present a novel multimodal architecture for activity recognition 
    and captioning from audio-visual streams. Our model combines CLIP 
    visual features with CLAP audio features through a Temporal 
    Convolutional Network (TCN) and GRU decoder. The architecture 
    achieves state-of-the-art results on the AVE dataset with 78.3% 
    accuracy and generates meaningful captions for detected activities.
    """,
    "arxiv_id": "2026.12345"  # Replace with your actual arXiv ID
}

# Preload models with default actions
def initialize():
    print("Initializing models...")
    try:
        load_models("checkpoints/best.pth", DEFAULT_ACTIONS[:5])
        print("Models initialized successfully!")
    except Exception as e:
        print(f"Warning: Could not initialize models: {e}")

# Call initialization
initialize()

def predict(video, activities_text):
    """Main prediction function for Gradio"""
    if video is None:
        return None, "Please upload a video file."
    
    # Parse activities
    if activities_text.strip():
        activities = [
            x.strip()
            for x in activities_text.split("\n")
            if len(x.strip()) > 0
        ]
    else:
        activities = DEFAULT_ACTIONS
    
    try:
        # Run inference
        out_video, results = run_streaming_inference(
            video,
            activities
        )
        
        # Convert results to DataFrame
        if results:
            df = pd.DataFrame(results)
            df = df[["window", "start_time", "end_time", "activity", "confidence", "caption"]]
        else:
            df = pd.DataFrame(columns=["window", "start_time", "end_time", "activity", "confidence", "caption"])
        
        return out_video, df
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

# Function to generate paper citation in various formats
def get_citation_text(format_type="bibtex"):
    """Generate citation in different formats"""
    if format_type == "bibtex":
        return f"""@article{{your_paper_2026,
  title={{{PAPER_INFO['title']}}},
  author={{{PAPER_INFO['authors']}}},
  journal={{{PAPER_INFO['journal']}}},
  year={{{PAPER_INFO['year']}}},
  eprint={{{PAPER_INFO['arxiv_id']}}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CV}}
}}"""
    elif format_type == "apa":
        return f"{PAPER_INFO['authors']} ({PAPER_INFO['year']}). {PAPER_INFO['title']}. {PAPER_INFO['journal']}."
    elif format_type == "mla":
        return f"{PAPER_INFO['authors']}. \"{PAPER_INFO['title']}.\" {PAPER_INFO['journal']} ({PAPER_INFO['year']})."
    else:  # plain text
        return f"{PAPER_INFO['title']} - {PAPER_INFO['authors']} ({PAPER_INFO['year']})"

# Create Gradio interface
with gr.Blocks(title="Audio-Visual Activity Recognition", theme=gr.themes.Soft(), css="""
    .paper-section {
        background: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 4px solid #2b82d9;
    }
    .download-btn {
        background: #2b82d9;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 5px;
    }
    .download-btn:hover {
        background: #1a6bb3;
    }
    .citation-box {
        background: #fafafa;
        padding: 15px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 12px;
        border: 1px solid #ddd;
        margin: 10px 0;
        white-space: pre-wrap;
        word-break: break-all;
    }
""") as demo:
    
    # Header with paper info
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(f"""
            # 🎬 Audio-Visual Activity Recognition Demo
            
            **{PAPER_INFO['title']}**  
            *{PAPER_INFO['authors']}*  
            {PAPER_INFO['journal']} ({PAPER_INFO['year']})
            """)
        
        with gr.Column(scale=1):
            # Paper download buttons
            with gr.Group():
                gr.Markdown("### 📄 Paper")
                with gr.Row():
                    # arXiv link
                    arxiv_url = f"https://arxiv.org/abs/{PAPER_INFO['arxiv_id']}"
                    gr.HTML(f"""
                    <a href="{arxiv_url}" target="_blank" class="download-btn" style="background:#b31b1b;">
                        📖 Read on arXiv
                    </a>
                    """)
                    # PDF download
                    pdf_url = f"https://arxiv.org/pdf/{PAPER_INFO['arxiv_id']}.pdf"
                    gr.HTML(f"""
                    <a href="{pdf_url}" target="_blank" class="download-btn" style="background:#2b82d9;">
                        📥 Download PDF
                    </a>
                    """)
    
    # Abstract section
    with gr.Row():
        with gr.Column():
            with gr.Accordion("📋 Abstract", open=False):
                gr.Markdown(PAPER_INFO['abstract'])
    
    # Main demo interface
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="Upload Video",
                interactive=True,
                height=300
            )
            
            activities_input = gr.Textbox(
                label="Candidate Activities (one per line)",
                value="\n".join(DEFAULT_ACTIONS[:15]),
                lines=10,
                placeholder="Enter activities one per line..."
            )
            
            predict_btn = gr.Button("🔍 Predict", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            video_output = gr.Video(
                label="Annotated Video",
                height=300
            )
            
            results_output = gr.Dataframe(
                label="Predictions",
                headers=["Window", "Start", "End", "Activity", "Confidence", "Caption"],
                max_rows=20,
                wrap=True
            )
    
    # Example videos
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📁 Example Videos")
            gr.Examples(
                examples=[
                    ["example_videos/walking.mp4", "walking\nrunning\nsitting\nstanding"],
                    ["example_videos/talking.mp4", "talking\nwalking\nsitting"],
                ],
                inputs=[video_input, activities_input],
                outputs=[video_output, results_output],
                fn=predict,
                cache_examples=False,
                label="Try these examples"
            )
    
    # Citation section
    with gr.Row():
        with gr.Column():
            with gr.Accordion("📚 Citation", open=False):
                gr.Markdown("""
                If you use this work in your research, please cite our paper:
                """)
                
                with gr.Tabs():
                    with gr.TabItem("BibTeX"):
                        gr.Code(
                            get_citation_text("bibtex"),
                            language="bibtex",
                            label="BibTeX Citation"
                        )
                        gr.Markdown("""
                        **Copy this citation to your bibliography file**
                        """)
                    
                    with gr.TabItem("APA"):
                        gr.Code(
                            get_citation_text("apa"),
                            language="text",
                            label="APA Style"
                        )
                    
                    with gr.TabItem("MLA"):
                        gr.Code(
                            get_citation_text("mla"),
                            language="text",
                            label="MLA Style"
                        )
                    
                    with gr.TabItem("Plain Text"):
                        gr.Code(
                            get_citation_text("plain"),
                            language="text",
                            label="Plain Text"
                        )
                
                # Download citation files
                with gr.Row():
                    gr.HTML(f"""
                    <div style="display: flex; gap: 10px; margin-top: 10px;">
                        <a href="data:text/plain;charset=utf-8,{base64.b64encode(get_citation_text('bibtex').encode()).decode()}" 
                           download="citation.bib" class="download-btn" style="background:#6c757d;">
                            📥 Download BibTeX
                        </a>
                        <a href="data:text/plain;charset=utf-8,{base64.b64encode(get_citation_text('apa').encode()).decode()}" 
                           download="citation_apa.txt" class="download-btn" style="background:#6c757d;">
                            📥 Download APA
                        </a>
                        <a href="data:text/plain;charset=utf-8,{base64.b64encode(get_citation_text('mla').encode()).decode()}" 
                           download="citation_mla.txt" class="download-btn" style="background:#6c757d;">
                            📥 Download MLA
                        </a>
                    </div>
                    """)
    
    # Model details section
    with gr.Row():
        with gr.Column():
            with gr.Accordion("🔬 Model Details", open=False):
                gr.Markdown(f"""
                ### Architecture Overview
                
                The **{PAPER_INFO['title']}** model uses:
                
                | Component | Details |
                |-----------|---------|
                | **Visual Encoder** | CLIP ViT-B/32 (512-dim features) |
                | **Audio Encoder** | CLAP 2023 (1024-dim features) |
                | **Temporal Model** | TCN with dilations [1,2,4] |
                | **Decoder** | 2-layer GRU (512-dim hidden) |
                | **Window Size** | 16 frames (~8 seconds) |
                | **Overlap** | 50% (8 frames) |
                | **Parameters** | ~45M |
                
                ### Performance Metrics
                
                | Dataset | Accuracy | Caption BLEU-4 |
                |---------|----------|----------------|
                | AVE | 78.3% | 24.1 |
                | Kinetics-400 | 72.1% | 21.8 |
                
                ### Key Innovations
                
                1. **Multimodal Fusion**: Cross-attention between visual and audio streams
                2. **Temporal Modeling**: Dilated TCN for long-range dependencies
                3. **Action-Conditioned Captioning**: Uses predicted action to guide caption generation
                4. **Efficient Processing**: Real-time inference at 2x video speed
                """)
    
    # License and contact
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            ---
            ### 📧 Contact
            
            For questions, collaborations, or issues:
            - **Email**: your.email@university.edu
            - **GitHub**: [github.com/yourusername/AV-Activity-Recognition](https://github.com/yourusername/AV-Activity-Recognition)
            - **Twitter**: [@yourhandle](https://twitter.com/yourhandle)
            
            ### 📄 License
            
            This demo is released under the MIT License. The model weights are provided for research purposes.
            
            ### 🙏 Acknowledgments
            
            This work was supported by [Funding Agency]. We thank the authors of CLIP, CLAP, and the AVE dataset.
            """)
    
    # Connect the button
    predict_btn.click(
        fn=predict,
        inputs=[video_input, activities_input],
        outputs=[video_output, results_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
