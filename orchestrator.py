import subprocess
import os
import sys
import argparse
import time

def run_command(command):
    """Executes a shell command and manages errors."""
    print(f"\n🚀 EXECUTING: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during execution of: {' '.join(command)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Orchestrator for Iterative Noisy Student with Ensemble")
    parser.add_argument("--start_gen", type=int, default=4, help="Starting generation")
    parser.add_argument("--num_loops", type=int, default=1, help="Number of complete cycles to execute")
    parser.add_argument("--teacher_enh", type=str, default="student_enhanced_gen_3", help="The Enhanced model to use as teacher")
    parser.add_argument("--teacher_deep", type=str, default="deep", help="The Deep model to use as teacher")
    args = parser.parse_args()

    print("🎸 --- DEEP CHORD AUTO-ORCHESTRATOR (ENSEMBLE PHASE) ---")
    print(f"Starting from Generation: {args.start_gen}")
    print(f"Teacher Enhanced: {args.teacher_enh}")
    print(f"Teacher Deep: {args.teacher_deep}")
    print("----------------------------------------")

    current_teacher_enh = args.teacher_enh

    for i in range(args.num_loops):
        gen = args.start_gen + i
        student_name = f"student_enhanced_gen_{gen}_ensemble"
        
        # 1. Confidence threshold: starting cautiously with the ensemble
        threshold = 0.75 + (i * 0.02)
        
        print(f"\n🌟 === START GENERATION CYCLE {gen} (ENSEMBLE) ===")
        print(f"Teachers: {current_teacher_enh} + {args.teacher_deep} | Target: {student_name} | Threshold: {threshold:.2f}")

        # STEP 1: Pseudo-Labels Generation with Ensemble
        label_cmd = [
            "python3", "src/training/generate_pseudo_labels.py",
            "--teacher_enh", current_teacher_enh,
            "--teacher_deep", args.teacher_deep,
            "--threshold", str(threshold)
        ]
        run_command(label_cmd)

        # STEP 2: Student Training
        train_cmd = [
            "python3", "train_student.py",
            "--teacher", current_teacher_enh, # Initialize with the best Enhanced weights
            "--student_name", student_name,
            "--epochs", "40"
        ]
        run_command(train_cmd)

        # STEP 3: Promotion
        print(f"✅ Cycle {gen} completed. {student_name} becomes the new Enhanced Teacher.")
        current_teacher_enh = student_name
        
        time.sleep(2) 

    print("\n🎉 [ORCHESTRATOR] All ensemble cycles completed!")

if __name__ == "__main__":
    main()
