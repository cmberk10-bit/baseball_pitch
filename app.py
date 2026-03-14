import tempfile
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from utils.config import OUTCOME_MODEL_PATH, INJURY_MODEL_PATH
from utils.feature_extraction import (
    extract_landmarks_from_image,
    extract_landmarks_from_video,
    compute_pitching_features,
    get_feature_vector_from_landmarks_sequence,
)
from utils.injury_risk import rule_based_injury_assessment, OPTIMAL_RANGES
from utils.visualization import draw_pose_on_image, create_feature_bar_chart, create_time_series_plot
from utils.gemini_coach import generate_coaching_plan

st.set_page_config(page_title="Baseball Pitching Intelligence", layout="wide")

st.title("Baseball Pitching Intelligence")
st.caption("Explainable pitcher mechanics analysis with injury-risk prevention")

outcome_model = joblib.load(OUTCOME_MODEL_PATH) if OUTCOME_MODEL_PATH.exists() else None
injury_model = joblib.load(INJURY_MODEL_PATH) if INJURY_MODEL_PATH.exists() else None

mode = st.sidebar.selectbox("Select Mode", ["Image Upload", "Video Upload", "Live Webcam"])


def predict_outcome(feature_dict):
    if outcome_model is None:
        return {"label": "Model not trained", "confidence": 0.0, "class_probabilities": {}}
    X = np.array([list(feature_dict.values())], dtype=float)
    probs = outcome_model.predict_proba(X)[0]
    pred = outcome_model.predict(X)[0]
    return {
        "label": pred,
        "confidence": float(np.max(probs)),
        "class_probabilities": dict(zip(outcome_model.classes_, probs.tolist()))
    }


def predict_injury_ml(feature_dict):
    if injury_model is None:
        return {"label": "Model not trained", "confidence": 0.0, "class_probabilities": {}}
    X = np.array([list(feature_dict.values())], dtype=float)
    probs = injury_model.predict_proba(X)[0]
    pred = injury_model.predict(X)[0]
    return {
        "label": pred,
        "confidence": float(np.max(probs)),
        "class_probabilities": dict(zip(injury_model.classes_, probs.tolist()))
    }


def render_ranges(metrics):
    rows = []
    for k, v in metrics.items():
        if k in OPTIMAL_RANGES:
            low, high = OPTIMAL_RANGES[k]
            color = "🟢" if low <= v <= high else ("🟡" if (low - 5) <= v <= (high + 5) else "🔴")
            rows.append([k, round(v, 2), f"{low} - {high}", color])
    if rows:
        st.subheader("Pitcher values vs target ranges")
        st.dataframe(pd.DataFrame(rows, columns=["Feature", "Value", "Target", "Status"]), use_container_width=True)


if mode == "Image Upload":
    uploaded = st.file_uploader("Upload pitcher image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        landmarks, pose_landmarks = extract_landmarks_from_image(image)
        if landmarks is None:
            st.error("No pitcher pose detected.")
        else:
            features = compute_pitching_features(landmarks)
            injury_rule = rule_based_injury_assessment(features)
            injury_ml = predict_injury_ml(features)
            outcome = predict_outcome(features)

            overlay = draw_pose_on_image(image, pose_landmarks)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB")

            c1, c2, c3 = st.columns(3)
            c1.metric("Delivery Score", outcome["label"])
            c2.metric("Confidence", f"{outcome['confidence']:.1%}")
            c3.metric("Injury Risk", injury_rule["overall_risk"])

            if injury_rule["warnings"]:
                st.write("### Risk warnings")
                for w in injury_rule["warnings"]:
                    st.warning(w)

            render_ranges(features)
            st.plotly_chart(create_feature_bar_chart(features, title="Pitching mechanics features"), use_container_width=True)

            with st.expander("Explainability drivers"):
                st.json({
                    "delivery_prediction": outcome,
                    "injury_rule_drivers": injury_rule["drivers"],
                    "injury_ml": injury_ml
                })

            if st.button("Generate Coaching Plan"):
                plan = generate_coaching_plan(features, injury_rule, outcome)
                st.write(plan)

elif mode == "Video Upload":
    uploaded = st.file_uploader("Upload pitcher video", type=["mp4", "mov", "avi"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            temp_path = tmp.name

        landmarks_seq, frames = extract_landmarks_from_video(temp_path, sample_rate=4)
        if not landmarks_seq:
            st.error("No pitcher pose detected in video.")
        else:
            agg_features, per_frame = get_feature_vector_from_landmarks_sequence(landmarks_seq)
            mean_features = {k.replace("_mean", ""): v for k, v in agg_features.items() if k.endswith("_mean")}

            injury_rule = rule_based_injury_assessment(mean_features)
            injury_ml = predict_injury_ml(mean_features)
            outcome = predict_outcome(mean_features)

            st.video(temp_path)

            c1, c2, c3 = st.columns(3)
            c1.metric("Delivery Score", outcome["label"])
            c2.metric("Confidence", f"{outcome['confidence']:.1%}")
            c3.metric("Injury Risk", injury_rule["overall_risk"])

            if injury_rule["warnings"]:
                for w in injury_rule["warnings"]:
                    st.warning(w)

            render_ranges(mean_features)

            feature_to_plot = st.selectbox("Select pitching metric over time", list(per_frame[0].keys()))
            st.plotly_chart(create_time_series_plot(per_frame, feature_to_plot), use_container_width=True)

            with st.expander("Explainability drivers"):
                st.json({
                    "delivery_prediction": outcome,
                    "injury_rule_drivers": injury_rule["drivers"],
                    "injury_ml": injury_ml
                })

            if st.button("Generate Coaching Plan"):
                plan = generate_coaching_plan(mean_features, injury_rule, outcome)
                st.write(plan)

else:
    st.info("Capture a pitcher frame and the app will analyze the delivery posture.")
    webcam = st.camera_input("Capture pitcher frame")
    if webcam:
        file_bytes = np.asarray(bytearray(webcam.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        landmarks, pose_landmarks = extract_landmarks_from_image(image)

        if landmarks is None:
            st.error("No pitcher pose detected.")
        else:
            features = compute_pitching_features(landmarks)
            injury_rule = rule_based_injury_assessment(features)
            outcome = predict_outcome(features)

            overlay = draw_pose_on_image(image, pose_landmarks)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB")

            st.success(f"Analyzing... Delivery: {outcome['label']} ({outcome['confidence']:.1%})")
            st.write("Injury Risk:", injury_rule["overall_risk"])
            render_ranges(features)
