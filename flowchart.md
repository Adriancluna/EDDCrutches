# CrutchGuard Application Flowchart

## User Flow Overview

```mermaid
flowchart TD
    subgraph Step1["Step 1: Body Measurements"]
        A[Start] --> B[Enter Height]
        B --> C[Enter Weight]
        C --> D[Select Unit System<br/>Imperial/Metric]
        D --> E[Continue]
    end

    subgraph Step2["Step 2: Crutch Configuration"]
        E --> F[Enter Grip Height Range<br/>Min/Max + Positions]
        F --> G[Enter Overall Length Range<br/>Min/Max + Positions]
        G --> H[Get AI Recommendation]
        H --> I{KNN Model<br/>Available?}
        I -->|Yes| J[KNN Hybrid Recommendation]
        I -->|No| K[Clinical Formula Fallback]
        J --> L[Display Recommended<br/>Grip & Overall Positions]
        K --> L
        L --> M[User Adjusts Crutches<br/>to Recommended Settings]
        M --> N[Continue to Calibration]
    end

    subgraph Step3["Step 3: Calibration"]
        N --> O[Start Camera Stream]
        O --> P[MediaPipe Pose Detection]
        P --> Q{All Body Parts<br/>Visible?}
        Q -->|No| R[Show Visibility Checklist<br/>Head/Arms/Legs/Distance]
        R --> P
        Q -->|Yes| S{Correct Distance?<br/>5-7 feet}
        S -->|No| T[Show Distance Warning]
        T --> P
        S -->|Yes| U[Hold Still for<br/>Calibration 30 frames]
        U --> V{Calibration<br/>Complete?}
        V -->|No| U
        V -->|Yes| W[Calculate pixels_per_cm<br/>from leg length]
        W --> X[Auto-advance to Step 4]
    end

    subgraph Step4["Step 4: Gait Analysis"]
        X --> Y[Start Analysis Mode]
        Y --> Z[Process Video Frames]
        Z --> AA[Detect Gait Phase<br/>Standing/Weight-Bearing/Swing]
        AA --> AB[Calculate Angles]
        AB --> AC[Right/Left Elbow Angles]
        AB --> AD[Right/Left Knee Angles]
        AB --> AE[Trunk Lean Angle]
        AB --> AF[Step Length & Base Width]
        AC --> AG[Evaluate Metrics]
        AD --> AG
        AE --> AG
        AF --> AG
        AG --> AH[Display Real-time Feedback]
        AH --> AI[Store Session Frames<br/>for Analysis]
        AI --> AJ{User Clicks<br/>Finish Trial?}
        AJ -->|No| Z
        AJ -->|Yes| AK[Stop Camera Stream]
        AK --> AL[Go to Step 5]
    end

    subgraph Step5["Step 5: Feedback & Results"]
        AL --> AM[Fetch Session Summary]
        AM --> AN[Analyze Biomechanics<br/>from Weight-Bearing Frames]
        AN --> AO[Generate Body Map<br/>Color-coded by status]
        AO --> AP[Generate Issues List<br/>with recommendations]
        AP --> AQ[Display Body Analysis<br/>Visualization]
        AQ --> AR[User Rates Experience<br/>1-5 scale]
        AR --> AS[User Reports Issues<br/>Checkboxes]
        AS --> AT[Submit Feedback]
        AT --> AU{Settings<br/>Optimal?}
        AU -->|Yes| AV[Show Success Message]
        AU -->|No| AW[Show New Recommended<br/>Settings]
        AV --> AX{Try New<br/>Settings?}
        AW --> AX
        AX -->|Yes| AY[Apply New Settings]
        AY --> AZ[Increment Trial Number]
        AZ --> BA[Reset Issue Tracking]
        BA --> N
        AX -->|No| BB[Finish Session]
        BB --> BC[Show Final Summary]
        BC --> BD[Return to Home]
    end
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph Frontend["Frontend (Browser)"]
        UI[assessment.html]
        JS[app.js]
        CSS[styles.css]
    end

    subgraph Backend["Backend (Flask)"]
        APP[app.py]
        subgraph Recommender["Recommender System"]
            KNN[knn_recommender.py]
            OPT[live_optimizer.py]
        end
        subgraph Core["Core Analysis"]
            GEOM[geometry.py]
            GAIT[gait_detector.py]
            EVAL[evaluators.py]
            TRACK[temporal_tracker.py]
            FILT[one_euro_filter.py]
        end
        subgraph Config["Configuration"]
            CRUTCH[crutch_models.py]
        end
    end

    subgraph External["External Libraries"]
        MP[MediaPipe Pose]
        CV[OpenCV]
        SK[scikit-learn]
    end

    UI <-->|WebSocket| APP
    JS <-->|REST API| APP
    APP --> KNN
    APP --> OPT
    APP --> GEOM
    APP --> GAIT
    APP --> EVAL
    APP --> TRACK
    APP --> FILT
    APP --> CRUTCH
    APP --> MP
    APP --> CV
    KNN --> SK
```

## API Endpoints

```mermaid
flowchart TD
    subgraph REST["REST API Endpoints"]
        POST1["POST /api/config<br/>Save user measurements"]
        GET1["GET /api/recommend<br/>Get AI crutch settings"]
        POST2["POST /api/start_calibration<br/>Begin calibration mode"]
        POST3["POST /api/start_analysis<br/>Begin gait analysis"]
        POST4["POST /api/new_trial<br/>Start new trial with settings"]
        GET2["GET /api/session_summary<br/>Get analysis results + body map"]
        POST5["POST /api/submit_feedback<br/>Process user feedback"]
    end

    subgraph WebSocket["WebSocket Events"]
        WS1["start_stream<br/>Begin video streaming"]
        WS2["stop_stream<br/>End video streaming"]
        WS3["video_frame<br/>Receive processed frame"]
        WS4["calibration_update<br/>Receive calibration status"]
        WS5["analysis_update<br/>Receive real-time metrics"]
    end
```

## Gait Phase Detection

```mermaid
stateDiagram-v2
    [*] --> STANDING: Initial
    STANDING --> WEIGHT_BEARING_LEFT: Left foot forward
    STANDING --> WEIGHT_BEARING_RIGHT: Right foot forward
    WEIGHT_BEARING_LEFT --> DOUBLE_SUPPORT: Both feet grounded
    WEIGHT_BEARING_RIGHT --> DOUBLE_SUPPORT: Both feet grounded
    DOUBLE_SUPPORT --> SWING_PHASE: One foot lifted
    SWING_PHASE --> WEIGHT_BEARING_LEFT: Left foot lands
    SWING_PHASE --> WEIGHT_BEARING_RIGHT: Right foot lands
    WEIGHT_BEARING_LEFT --> STANDING: Feet together
    WEIGHT_BEARING_RIGHT --> STANDING: Feet together
```

## Body Map Status Logic

```mermaid
flowchart TD
    subgraph ElbowAnalysis["Elbow Angle Analysis"]
        E1[Measure Elbow Angle] --> E2{Angle < 140°?}
        E2 -->|Yes| E3[CRITICAL: Too Bent<br/>Raise grip height]
        E2 -->|No| E4{Angle > 170°?}
        E4 -->|Yes| E5[CRITICAL: Too Straight<br/>Lower grip height]
        E4 -->|No| E6{140-150° or 160-170°?}
        E6 -->|Yes| E7[WARNING: Slight adjustment needed]
        E6 -->|No| E8[GOOD: Optimal range 150-160°]
    end

    subgraph TrunkAnalysis["Trunk Lean Analysis"]
        T1[Measure Trunk Lean] --> T2{Lean > 15°?}
        T2 -->|Yes| T3[CRITICAL: Excessive lean<br/>Increase overall length]
        T2 -->|No| T4{Lean > 8°?}
        T4 -->|Yes| T5[WARNING: Slight lean]
        T4 -->|No| T6[GOOD: Upright posture]
    end

    subgraph AsymmetryAnalysis["Asymmetry Analysis"]
        A1[Compare L/R Elbows] --> A2{Difference > 10°?}
        A2 -->|Yes| A3[CRITICAL: Significant asymmetry<br/>Distribute weight evenly]
        A2 -->|No| A4{Difference > 5°?}
        A4 -->|Yes| A5[WARNING: Mild asymmetry]
        A4 -->|No| A6[GOOD: Balanced]
    end
```

## Recommendation System Flow

```mermaid
flowchart TD
    START[User Height + Crutch Specs] --> CHECK{Height in<br/>KNN training range?}
    CHECK -->|Yes 150-190cm| KNN[KNN Model Prediction]
    CHECK -->|No| CLINICAL[Clinical Formula]

    KNN --> HYBRID{Confidence<br/>High?}
    HYBRID -->|Yes > 0.8| USE_KNN[Use KNN Result]
    HYBRID -->|No| BLEND[Blend KNN + Clinical<br/>Weighted Average]

    CLINICAL --> GRIP_CALC[Grip = Wrist Height<br/>~82% of height]
    CLINICAL --> OVERALL_CALC[Overall = Height × 0.77]

    USE_KNN --> SNAP[Snap to Nearest<br/>Crutch Position]
    BLEND --> SNAP
    GRIP_CALC --> SNAP
    OVERALL_CALC --> SNAP

    SNAP --> OUTPUT[Recommended Settings<br/>+ Confidence Score]
```

---

## File Structure

```
crutch_gait_analysis/
├── app.py                      # Main Flask application
├── frontend/
│   ├── index.html              # Landing page
│   ├── assessment.html         # 5-step assessment flow
│   ├── css/
│   │   └── styles.css          # Global styles
│   └── js/
│       └── app.js              # Frontend JavaScript
├── backend/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── crutch_models.py    # Crutch device profiles
│   ├── core/
│   │   ├── __init__.py
│   │   ├── geometry.py         # Angle/distance calculations
│   │   ├── gait_detector.py    # Gait phase detection
│   │   ├── evaluators.py       # Metric evaluation functions
│   │   ├── temporal_tracker.py # Moving average tracking
│   │   └── one_euro_filter.py  # Noise filtering
│   └── recommender/
│       ├── __init__.py
│       ├── knn_recommender.py  # KNN-based recommendations
│       └── live_optimizer.py   # Real-time optimization
├── models/
│   └── knn_crutch_model.joblib # Trained KNN model
└── misc/
    └── flowchart.md            # This file
```

## Key Technologies

| Component | Technology |
|-----------|------------|
| Backend Server | Flask + Flask-SocketIO |
| Pose Estimation | MediaPipe Pose |
| Video Processing | OpenCV |
| ML Model | scikit-learn KNN |
| Real-time Communication | WebSocket |
| Frontend | Vanilla JavaScript |
| Styling | CSS Custom Properties |
