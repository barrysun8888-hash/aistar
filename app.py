"""Streamlit app for Mirror AI - 智能形象进化系统."""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

from mirror_ai.face_analysis import (
    FaceLandmarks,
    FaceShapeClassifier,
    FeatureExtractor,
    SkinToneAnalyzer,
    ExpressionAnalyzer,
)
from mirror_ai.recommendation import MakeupRecommender, HairstyleRecommender, LightingRecommender, StyleProfiler
from mirror_ai.preview import VirtualTryOn
from mirror_ai.user_profile import UserProfile


# Page config
st.set_page_config(
    page_title="镜AI - 智能形象进化系统",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "landmarks" not in st.session_state:
        st.session_state.landmarks = None
    if "image" not in st.session_state:
        st.session_state.image = None
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None


@st.cache_resource
def load_models():
    """Load ML models (cached for performance)."""
    try:
        landmarks_extractor = FaceLandmarks()
        face_shape_classifier = FaceShapeClassifier()
        feature_extractor = FeatureExtractor()
        skin_tone_analyzer = SkinToneAnalyzer()
        expression_analyzer = ExpressionAnalyzer()
        makeup_recommender = MakeupRecommender()
        hairstyle_recommender = HairstyleRecommender()
        lighting_recommender = LightingRecommender()
        style_profiler = StyleProfiler()
        virtual_tryon = VirtualTryOn()

        return {
            "landmarks": landmarks_extractor,
            "face_shape": face_shape_classifier,
            "features": feature_extractor,
            "skin_tone": skin_tone_analyzer,
            "expression": expression_analyzer,
            "makeup": makeup_recommender,
            "hairstyle": hairstyle_recommender,
            "lighting": lighting_recommender,
            "style": style_profiler,
            "virtual_tryon": virtual_tryon,
        }
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None


def analyze_face(image: np.ndarray, models: dict) -> dict:
    """Perform full face analysis."""
    results = {}

    # Extract landmarks
    landmarks = models["landmarks"].extract(image)
    if landmarks is None:
        return {"error": "未检测到人脸，请上传清晰的正面照片"}

    results["landmarks"] = landmarks

    # Face shape
    face_shape = models["face_shape"].classify(landmarks, image.shape)
    results["face_shape"] = face_shape

    # Golden ratio analysis
    golden_ratio = models["face_shape"].get_golden_ratio_analysis(landmarks, image.shape)
    results["golden_ratio"] = golden_ratio

    # Features
    features = models["features"].extract_all(landmarks, image.shape)
    results["features"] = features

    # Skin tone
    skin_tone = models["skin_tone"].analyze(image, landmarks)
    results["skin_tone"] = skin_tone

    # Expression
    expression = models["expression"].analyze(landmarks)
    results["expression"] = expression

    return results


def main():
    init_session_state()

    # Header
    st.markdown('<p class="main-header">🪞 镜AI 智能形象进化系统</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">不是告诉你"像哪个明星"，而是告诉你"如何成为更好的自己"</p>', unsafe_allow_html=True)

    # Load models
    models = load_models()
    if models is None:
        st.stop()

    # Main content
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📤 上传照片")

        uploaded_file = st.file_uploader(
            "选择一张清晰的正面照片",
            type=["jpg", "jpeg", "png"],
            help="请上传清晰的正面照，避免侧脸、遮挡或模糊照片"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            image = np.array(image)

            # Convert to BGR if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            st.session_state.image = image

            # Display uploaded image
            display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(display_img, caption="上传的照片", use_container_width=True)

        if st.session_state.image is not None and st.button("🔍 开始分析", type="primary"):
            with st.spinner("分析中..."):
                results = analyze_face(st.session_state.image, models)

                if "error" in results:
                    st.error(results["error"])
                else:
                    st.session_state.analysis_results = results
                    st.session_state.analysis_complete = True
                    st.success("分析完成！")
                    st.rerun()

    with col2:
        if st.session_state.analysis_complete and st.session_state.analysis_results:
            results = st.session_state.analysis_results

            st.subheader("📊 分析报告")

            # Face Shape
            face_shape = results.get("face_shape", "unknown")
            col_shape, col_skin = st.columns(2)

            with col_shape:
                st.metric("脸型", face_shape)

            with col_skin:
                skin_tone = results.get("skin_tone", {})
                st.metric("肤色", f"Type {skin_tone.get('shade_level', '?')}")

            # Golden Ratio
            golden = results.get("golden_ratio", {})
            if golden.get("is_ideal_proportion"):
                st.success("✓ 面部比例接近黄金比例")
            else:
                st.info(f"面部比例偏离黄金比例 {golden.get('deviation_from_golden', 0):.2f}")

            # Features
            features = results.get("features", {})
            st.markdown("### 👁 五官特征")

            eye_type = features.get("eyes", {}).get("type", "unknown")
            lip_fullness = features.get("lips", {}).get("fullness", "unknown")
            brow_info = features.get("eyebrows", {})

            col_eye, col_lip, col_brow = st.columns(3)
            with col_eye:
                st.write(f"**眼型**: {eye_type}")
            with col_lip:
                st.write(f"**唇形**: {lip_fullness}")
            with col_brow:
                st.write(f"**眉形**: {brow_info.get('thickness', 'unknown')}")

            # Expression
            expression = results.get("expression", {})
            st.markdown(f"**表情特征**: {expression.get('classified_expression', 'neutral')}")

            # Skin tone details
            if skin_tone:
                st.markdown("### 🎨 肤色分析")
                st.write(f"**Undertone**: {skin_tone.get('undertone', 'unknown')}")
                st.write(f"**肤色等级**: {skin_tone.get('shade_name', 'unknown')}")

                recommendations = skin_tone.get("color_recommendations", {})
                if recommendations:
                    st.write("**推荐口红色**: {', '.join(recommendations.get('lipstick', [])[:3])}")

            # Makeup Recommendations
            st.markdown("---")
            st.markdown("### 💄 妆容建议")

            makeup_recs = models["makeup"].recommend(
                face_shape=face_shape,
                features=features,
                skin_tone=skin_tone,
            )

            # Display recommendations
            base_recs = makeup_recs.get("base_recommendations", {})

            tab1, tab2 = st.tabs(["优化轨", "突破轨"])

            with tab1:
                optimize = makeup_recs.get("optimize_track", {})
                st.markdown(f"**策略**: {optimize.get('strategy', '')}")
                st.markdown(f"**重点**: {optimize.get('focus_advice', '')}")

            with tab2:
                breakthrough = makeup_recs.get("breakthrough_track", {})
                st.markdown(f"**策略**: {breakthrough.get('strategy', '')}")
                st.markdown(f"**建议**: {breakthrough.get('crossover_tip', '')}")

            # Virtual Try-on
            st.markdown("---")
            st.markdown("### ✨ 虚拟试妆")

            landmarks = results.get("landmarks")
            if landmarks is not None:
                # Sample makeup settings
                makeup_settings = {
                    "lipstick": {
                        "color": (180, 80, 100),  # Coral
                        "opacity": 0.8,
                    },
                    "eyeshadow": {
                        "color": (120, 80, 60),  # Brown
                        "liner": (30, 30, 30),
                        "intensity": 0.6,
                    },
                    "blush": {
                        "color": (200, 120, 120),
                        "intensity": 0.3,
                    },
                    "highlight": {
                        "enabled": True,
                        "intensity": 0.4,
                    },
                    "contour": {
                        "face_shape": face_shape,
                        "intensity": 0.4,
                    },
                }

                before, after = models["virtual_tryon"].generate_before_after(
                    st.session_state.image, landmarks, makeup_settings
                )

                # Display comparison
                comparison = models["virtual_tryon"].create_comparison_grid(before, after, labels=True)

                st.image(
                    cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB),
                    caption="虚拟试妆效果",
                    use_container_width=True
                )

                st.caption("提示：这是基于你面部特征的虚拟预览，实际效果可能有所不同")

            # Hairstyle Recommendations
            st.markdown("---")
            st.markdown("### 💇 发型建议")

            hair_recs = models["hairstyle"].recommend(
                face_shape=face_shape,
                forehead_height="normal",
                hair_texture="medium",
            )

            st.write("**推荐长度**: " + " | ".join(hair_recs.get("recommended_lengths", [])[:2]))

            bangs = hair_recs.get("bangs_advice", {})
            st.write(f"**刘海建议**: {bangs.get('shape_advice', '')}")

            # Lighting Recommendations
            st.markdown("---")
            st.markdown("### 💡 光影建议")

            light_recs = models["lighting"].recommend(face_shape=face_shape, skin_tone=skin_tone)
            primary = light_recs.get("primary_lighting", {})

            st.write(f"**推荐布光**: {primary.get('name', '侧光45度')}")
            st.write(f"**效果**: {primary.get('effect', '')}")

            # Style Profile
            st.markdown("---")
            st.markdown("### 🎭 风格定位")

            style_profile = models["style"].profile(
                skin_tone=skin_tone,
                features=features,
                expression=expression,
            )

            season = style_profile.get("color_season", "spring")
            st.metric("色彩季型", style_profile.get("season_name", "春季型"))

            st.write("**特点**: " + "、".join(style_profile.get("characteristics", [])))

            # Color palette
            palette = style_profile.get("color_palette", {})
            if palette:
                st.write("**主推色**: " + "、".join(palette.get("primary", [])[:3]))

        else:
            # Empty state
            st.info("👈 请上传照片并点击「开始分析」")
            st.markdown("""
            ### 📋 使用流程

            1. **上传照片** - 清晰的正面照
            2. **AI分析** - 面部特征、肤色、比例
            3. **获取建议** - 妆容、发型、光影
            4. **虚拟试妆** - 预览改造效果
            5. **实践反馈** - 上传成果照获得优化建议
            """)


if __name__ == "__main__":
    main()
