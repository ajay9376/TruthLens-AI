import streamlit as st
import os
import sys
import tempfile

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- CUSTOM CSS --------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
    }
    .result-real {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .result-fake {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .result-suspicious {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .score-card {
        background: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("""
<div class="main-header">
    <h1>🔍 TruthLens AI</h1>
    <p>Advanced Deepfake Detection System</p>
    <p>Powered by SyncNet • Face Texture • Blink Analysis • Lip Reading</p>
</div>
""", unsafe_allow_html=True)

# -------- SIDEBAR --------
st.sidebar.image("https://via.placeholder.com/300x100/667eea/white?text=TruthLens+AI")
st.sidebar.markdown("## 🎯 About")
st.sidebar.markdown("""
TruthLens AI uses **4 powerful signals** to detect deepfakes:

- 🎵 **SyncNet** — Lip sync analysis
- 🎨 **Face Texture** — Skin texture analysis  
- 👁️ **Blink Pattern** — Eye blink detection
- 👄 **Lip Reader** — Speech articulation
""")

st.sidebar.markdown("## ⚙️ Settings")
show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

st.sidebar.markdown("## 📊 Thresholds")
st.sidebar.markdown("""
- ✅ **REAL**: Score ≥ 60
- ⚠️ **SUSPICIOUS**: Score 40-60  
- ❌ **DEEPFAKE**: Score < 40
""")

# -------- MAIN CONTENT --------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("## 📹 Upload Video")
    uploaded_file = st.file_uploader(
        "Upload a video to analyze",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload any video file to check if it's a deepfake"
    )

    if uploaded_file:
        st.video(uploaded_file)
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        st.info(f"📦 Size: {uploaded_file.size / 1024 / 1024:.2f} MB")

with col2:
    st.markdown("## 🔍 Analysis")

    if uploaded_file:
        if st.button("🚀 Analyze Video", type="primary", use_container_width=True):

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Run analysis
            with st.spinner("🤖 TruthLens AI is analyzing..."):

                # Import combined detector
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, BASE_DIR)

                try:
                    from combined_detector import detect
                    results = detect(tmp_path)

                    if results:
                        # ---- VERDICT ----
                        verdict = results['verdict']
                        final_score = results['final_score']

                        if verdict == "REAL":
                            st.markdown(f"""
                            <div class="result-real">
                                ✅ REAL VIDEO<br>
                                <small>Confidence: {final_score:.1f}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                        elif verdict == "SUSPICIOUS":
                            st.markdown(f"""
                            <div class="result-suspicious">
                                ⚠️ SUSPICIOUS<br>
                                <small>Confidence: {final_score:.1f}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-fake">
                                ❌ DEEPFAKE DETECTED<br>
                                <small>Confidence: {final_score:.1f}%</small>
                            </div>
                            """, unsafe_allow_html=True)

                        # ---- SCORE BREAKDOWN ----
                        st.markdown("### 📊 Signal Breakdown")

                        signals = {
                            "🎵 SyncNet Lip-Sync": results['syncnet_score'],
                            "🎨 Face Texture": results['texture_score'],
                            "👁️ Blink Pattern": results['blink_score'],
                            "👄 Lip Reader": results['lip_score'],
                        }

                        for signal, score in signals.items():
                            col_s1, col_s2 = st.columns([3, 1])
                            with col_s1:
                                st.progress(int(score))
                                st.caption(signal)
                            with col_s2:
                                color = "🟢" if score >= 60 else "🟡" if score >= 40 else "🔴"
                                st.markdown(f"**{color} {score:.1f}**")

                        # ---- COMBINED SCORE ----
                        st.markdown("### 🎯 Combined Score")
                        st.progress(int(final_score))
                        st.markdown(f"**{final_score:.1f}/100**")

                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    if show_debug:
                        st.exception(e)

            # Cleanup
            os.unlink(tmp_path)
    else:
        st.info("👆 Upload a video to start analysis!")
        st.markdown("""
        ### 🎯 What TruthLens AI Detects:
        - Face swap deepfakes
        - AI generated videos
        - Voice cloned videos
        - Manipulated footage
        """)

# -------- FOOTER --------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🔍 TruthLens AI — Built with ❤️ | Powered by SyncNet, MediaPipe & YOLOv8</p>
</div>
""", unsafe_allow_html=True)