import subprocess
import os
import sys
import argparse
import time
import glob

def run_command(command):
    print(f"\n🚀 EXECUTING: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {' '.join(command)}")
        sys.exit(1)

def get_gold_track_count():
    """Conta i brani Gold disponibili per calcolare la proporzione Silver."""
    # Heuristic count
    billboard_files = glob.glob("data/processed/billboard/*.pt")
    guitarset_files = glob.glob("data/processed/guitarset/*.pt")
    return len(billboard_files) + len(guitarset_files)

def main():
    parser = argparse.ArgumentParser(description="Orchestrator with Smart Stochastic Labeling")
    parser.add_argument("--start_gen", type=int, default=7, help="Generation to start from")
    parser.add_argument("--num_loops", type=int, default=1, help="Number of full cycles")
    parser.add_argument("--teacher_enh", type=str, default="student_enhanced_gen_6", help="Enhanced teacher")
    parser.add_argument("--teacher_deep", type=str, default="super_master_deep_balanced", help="Deep teacher")
    args = parser.parse_args()

    print("🎸 --- DEEP CHORD AUTO-ORCHESTRATOR (SMART STOCHASTIC) ---")
    
    current_teacher_enh = args.teacher_enh
    gold_count = get_gold_track_count()
    
    # Calcolo dei brani Silver necessari per soddisfare il 70/30 (con margine di sicurezza 1.5x)
    # Proporzione: gold_count / 0.70 * 0.30
    target_silver_tracks = int((gold_count / 0.70) * 0.30 * 1.5)
    
    print(f"📊 Balanced Analysis: Gold Tracks: {gold_count} | Target Silver Tracks: {target_silver_tracks}")

    for i in range(args.num_loops):
        gen = args.start_gen + i
        student_name = f"student_enhanced_gen_{gen}_elite"
        threshold = 0.80
        
        print(f"\n🌟 === START CYCLE GENERATION {gen} (STOCHASTIC ELITE) ===")
        
        # STEP 1: Smart Labeling (Solo i brani necessari!)
        label_cmd = [
            "python3", "src/training/generate_pseudo_labels.py",
            "--teacher_enh", current_teacher_enh,
            "--teacher_deep", args.teacher_deep,
            "--threshold", str(threshold),
            "--max_tracks", str(target_silver_tracks)
        ]
        run_command(label_cmd)

        # STEP 2: Training
        train_cmd = [
            "python3", "train_student.py",
            "--teacher", current_teacher_enh, 
            "--student_name", student_name,
            "--epochs", "40"
        ]
        run_command(train_cmd)

        print(f"✅ Cycle {gen} completed. {student_name} is the new champion.")
        current_teacher_enh = student_name
        time.sleep(2) 

    print("\n🎉 [ORCHESTRATOR] Stochastic training completed successfully!")

if __name__ == "__main__":
    main()
