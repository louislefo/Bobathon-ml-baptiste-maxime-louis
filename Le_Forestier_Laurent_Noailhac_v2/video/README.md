# Video Deliverable

**Team**: Louis Le Forestier, Maxime Laurent, Baptiste Noailhac (ESILV - Data & AI Major)  
**Project**: Bobathon ESILV (Probabl x IBM)

---

## 1. Presentation Video
The official presentation video file is included directly in this directory:
- **Filename**: `Presentation.mp4`
- **File size**: 68.2 MB
- **Format**: MP4 (H.264 video, AAC audio)
- **Status**: Ready for evaluation

---

## 2. Presentation Outline & Coverage
The video presentation covers the key evaluation dimensions:
1. **Medical Context & Challenge (0:00 - 1:15)**: Predicting true OFF MDS-UPDRS Part III motor scores, understanding observational confounding (rater subjectivity, levodopa wearing-off, informative missingness, and longitudinal disease progression).
2. **Methodological Evolution (1:15 - 1:45)**: Progression from the 16.50 dummy baseline to linear models, gradient boosting trees, trajectory regression, and the champion Model 34 (~3.38 RMSE).
3. **Data Leakage & Grouped Validation (1:45 - 2:10)**: Preventing patient-level leakage across longitudinal visits using strict `GroupKFold` splits.
4. **IBM Bob Toolchain Usage (2:10 - 2:40)**: End-to-end development, automated test generation, multi-file code awareness, and high-velocity iteration (30+ model versions tested).
5. **Key Takeaways & Productivity Impact (2:40 - 3:05)**: Productivity boost, focusing energy on medical logic and model architectures.
