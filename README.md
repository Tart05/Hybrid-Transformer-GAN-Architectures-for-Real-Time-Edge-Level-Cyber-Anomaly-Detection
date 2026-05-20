# Hybrid-Transformer-GAN-Architectures-for-Real-Time-Edge-Level-Cyber-Anomaly-Detection
Baseline hybrid model for anamoly detection

╔══════════════════════════════════════════════════════════════════╗
║  SETUP GUIDE — Baseline Transformer-GAN (Teacher Model)        ║
║  Path: C:\Users\adith\Downloads\Design Project - Baseline      ║
╚══════════════════════════════════════════════════════════════════╝

Follow these steps to establish the high-performance Teacher baseline.

═══════════════════════════════════════════════════════════════════
STEP 1: ANACONDA ENVIRONMENT SETUP
═══════════════════════════════════════════════════════════════════

1. Open Anaconda Prompt.
2. Create the baseline environment:
    conda create -n baseline_gan python=3.10 -y

3. Activate the environment:
    conda activate baseline_gan

4. Install dependencies:
    pip install torch pandas scikit-learn numpy

═══════════════════════════════════════════════════════════════════
STEP 2: DIRECTORY VERIFICATION
═══════════════════════════════════════════════════════════════════

Ensure your folder structure matches exactly:
C:\Users\adith\Downloads\Design Project - Baseline\
    ├── data/
    │   ├── mqttdataset_reduced.csv
    │   └── test30_reduced.csv
    ├── data_loader.py
    ├── models.py
    ├── train_baseline.py
    └── demo_baseline.py

═══════════════════════════════════════════════════════════════════
STEP 3: RUNNING THE BASELINE
═══════════════════════════════════════════════════════════════════

1. Navigate to the folder:
    cd "C:\Users\adith\Downloads\Design Project - Baseline"

2. Start training the Teacher model:
    python train_baseline.py

3. Run the evaluation demo:
    python demo_baseline.py

NOTE: This will generate 'teacher_baseline.pt'. You MUST copy this 
file to your Optimized folder once training is complete.
═══════════════════════════════════════════════════════════════════
