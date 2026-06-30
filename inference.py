import os
import cv2
import torch
import clip
import librosa
import tempfile
import soundfile as sf
import numpy as np
from PIL import Image
from transformers import GPT2Tokenizer
from msclap import CLAP
import time

from model import AVTCNGRUModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables for caching
_clip_model = None
_clap_model = None
_tokenizer = None
_network = None
_action_bank = None

def load_models(model_path, actions):
    """Load all models once and cache them globally"""
    global _clip_model, _clap_model, _tokenizer, _network, _action_bank
    
    print("Loading models...")
    
    # Tokenizer
    _tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2", 
        local_files_only=True,
        cache_dir="./gpt2"
    )
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.add_special_tokens({'bos_token': '<BOS>'})
    
    # CLIP
    _clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    _clip_model.eval()
    
    # CLAP
    _clap_model = CLAP(version="2023", use_cuda=(DEVICE == "cuda"))
    
    # Network
    _network = AVTCNGRUModel(len(_tokenizer)).to(DEVICE)
    _network.load_state_dict(torch.load(model_path, map_location=DEVICE))
    _network.eval()
    
    # Build action text bank
    _action_bank = build_action_text_bank(actions)
    
    # Warmup
    dummy_v = torch.randn(1, 16, 512).to(DEVICE)
    dummy_a = torch.randn(1, 16, 1024).to(DEVICE)
    for _ in range(5):
        with torch.no_grad():
            _ = _network(dummy_v, dummy_a, _action_bank)
    
    print("Models loaded and warmed up!")
    return _clip_model, _clap_model, _tokenizer, _network, _action_bank

def build_action_text_bank(actions):
    """Build CLIP text embeddings for candidate actions"""
    global _clip_model
    tokens = clip.tokenize(actions).to(DEVICE)
    with torch.no_grad():
        emb = _clip_model.encode_text(tokens)
    return (emb / emb.norm(dim=-1, keepdim=True)).float()

def generate_caption(caption_emb, max_len=30):
    """Generate caption from caption embedding"""
    global _tokenizer, _network
    
    tokens = [_tokenizer.bos_token_id]
    
    for _ in range(max_len):
        inp = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = _network.decoder(caption_emb, inp)
        next_token = logits[:, -1].argmax(-1).item()
        tokens.append(next_token)
        
        if next_token == _tokenizer.eos_token_id:
            break
    
    return _tokenizer.decode(tokens, skip_special_tokens=True)

def wrap_text(text, max_words_per_line=6):
    """Wrap text for display"""
    words = text.split()
    lines = []
    for i in range(0, len(words), max_words_per_line):
        lines.append(" ".join(words[i:i+max_words_per_line]))
    return lines[:2]  # limit to 2 lines

def process_audio_chunk(chunk, sample_rate=22050):
    """Process audio chunk with CLAP"""
    global _clap_model
    
    if np.all(chunk == 0):
        return np.zeros(1024, dtype=np.float32)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, chunk, sample_rate)
        emb = _clap_model.get_audio_embeddings([f.name])[0]
    os.remove(f.name)
    
    if hasattr(emb, "cpu"):
        emb = emb.cpu().numpy()
    
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    
    # Ensure 1024 dimensions
    if emb.shape[0] == 512:
        emb = np.concatenate([emb, emb])
    
    return emb

def run_streaming_inference(video_path, actions):
    """
    Run inference on video with given candidate actions
    Returns: (output_video_path, results_table)
    """
    global _clip_model, _clap_model, _tokenizer, _network, _action_bank
    
    # Load models if not already loaded
    if _network is None:
        load_models("checkpoints/best.pth", actions)
    
    # Ensure action bank matches current actions
    # Simple check: if actions changed, rebuild
    if _action_bank is None or len(actions) != _action_bank.shape[0]:
        _action_bank = build_action_text_bank(actions)
    
    # Parameters
    SAMPLE_RATE = 22050
    CHUNK_SEC = 0.5
    WINDOW = 16
    
    # Video capture
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # fallback
    
    # Output video writer
    out_path = "output_stream_test.mp4"
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (300, 300)
    )
    
    # Load audio
    try:
        audio, _ = librosa.load(video_path, sr=SAMPLE_RATE)
        step = int(SAMPLE_RATE * CHUNK_SEC)
        audio_chunks = [audio[i:i+step] for i in range(0, len(audio), step)]
        has_audio = True
    except Exception as e:
        print(f"No audio found: {e}")
        audio_chunks = []
        has_audio = False
    
    frames_buffer = []
    audio_buffer = []
    results = []
    frame_idx = 0
    chunk_idx = 0
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frame every window step
        if frame_idx % int(fps * CHUNK_SEC) == 0:
            frame = cv2.resize(frame, (300, 300))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames_buffer.append(img)
            
            if chunk_idx < len(audio_chunks):
                audio_buffer.append(audio_chunks[chunk_idx])
                chunk_idx += 1
            else:
                # Dummy audio if not enough chunks
                dummy_chunk = np.zeros(int(SAMPLE_RATE * CHUNK_SEC), dtype=np.float32)
                audio_buffer.append(dummy_chunk)
        
        frame_idx += 1
        
        # Process window
        if len(frames_buffer) == WINDOW:
            v_feats, a_feats = [], []
            
            # Encode frames with CLIP
            for f in frames_buffer:
                inp = _clip_model.preprocess(f).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    emb = _clip_model.encode_image(inp)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                v_feats.append(emb.squeeze(0))
            
            # Encode audio with CLAP
            for chunk in audio_buffer:
                emb = process_audio_chunk(chunk, SAMPLE_RATE)
                a_feats.append(torch.tensor(emb))
            
            v_feats = torch.stack(v_feats).unsqueeze(0).float().to(DEVICE)
            a_feats = torch.stack(a_feats).unsqueeze(0).float().to(DEVICE)
            
            # Model prediction
            with torch.no_grad():
                logits_cls, caption_emb = _network(v_feats, a_feats, _action_bank)
            
            # Get predictions
            probs = torch.softmax(logits_cls, dim=1)
            action_idx = logits_cls.argmax().item()
            confidence = probs[0, action_idx].item()
            action = actions[action_idx]
            caption = generate_caption(caption_emb)
            
            # Store result
            timestamp = frame_idx / fps
            window_start = max(0, timestamp - WINDOW * CHUNK_SEC)
            results.append({
                "window": len(results) + 1,
                "start_time": round(window_start, 2),
                "end_time": round(timestamp, 2),
                "activity": action,
                "confidence": round(confidence, 3),
                "caption": caption
            })
            
            print(f"[Window {len(results)}] {action} ({confidence:.3f}) | {caption}")
            
            # Write annotated frames
            for f in frames_buffer:
                frame_np = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
                
                # Draw activity
                cv2.putText(frame_np, f"Activity:",
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0,255,0), 2)
                lines = wrap_text(action, max_words_per_line=6)
                y = 40
                for line in lines:
                    cv2.putText(frame_np, line,
                               (40, y), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (0,0,255), 2)
                    y += 30
                
                # Draw caption
                cv2.putText(frame_np, "Caption:",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0,255,0), 2)
                lines = wrap_text(caption, max_words_per_line=6)
                y = 90
                for line in lines:
                    cv2.putText(frame_np, line,
                               (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (0,0,255), 2)
                    y += 30
                
                # Draw confidence
                cv2.putText(frame_np, f"Conf: {confidence:.2f}",
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (255,255,0), 1)
                
                # Draw window info
                cv2.putText(frame_np, f"Window {len(results)}",
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (255,255,255), 1)
                
                out.write(frame_np)
            
            # Slide window with overlap
            frames_buffer = frames_buffer[8:]
            audio_buffer = audio_buffer[8:]
    
    # Cleanup
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    print(f"Processing time: {processing_time:.2f}s")
    
    return out_path, results
