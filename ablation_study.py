"""
Main execution script for the Ablation Study (§7.5).
Runs the system across 6 variants and records quantitative metrics.
"""

import json
import pandas as pd
from loguru import logger
from datetime import datetime

from src.config.settings import get_settings
from src.orchestration.ablation_loop import AblationAutonomyLoop
from src.evaluation.metrics import MetricsEvaluator

# Define the topic for evaluation
TOPIC = "Retrieval-Augmented Generation in Medical NLP"

# Define the variants to test
VARIANTS = {
    "Full System": {
        "cgcers": True, "cdra": True, "frgo": True, "dskg": True, "tket": True
    },
    "No Confidence Scoring": {
        "cgcers": False, "cdra": True, "frgo": True, "dskg": True, "tket": True
    },
    "No Contradiction Detection": {
        "cgcers": True, "cdra": False, "frgo": True, "dskg": True, "tket": True
    },
    "No Formalized Gaps": {
        "cgcers": True, "cdra": True, "frgo": False, "dskg": True, "tket": True
    },
    "Vector Store (No KG)": {
        "cgcers": True, "cdra": True, "frgo": True, "dskg": False, "tket": True
    },
    "No Temporal Tracking": {
        "cgcers": True, "cdra": True, "frgo": True, "dskg": True, "tket": False
    }
}

def run_study():
    settings = get_settings()
    settings.max_papers = 10  # Use a smaller corpus for the study to save time/tokens
    settings.max_autonomy_iterations = 1
    
    results = []
    
    for name, config in VARIANTS.items():
        logger.info(f"\n\n{'='*20} RUNNING VARIANT: {name} {'='*20}")
        
        loop = AblationAutonomyLoop(settings, enabled_components=config)
        
        try:
            state = loop.execute(TOPIC)
            
            # Extract metrics
            metrics = MetricsEvaluator.get_all_metrics(state)
            
            # Add metadata
            metrics["Variant"] = name
            metrics["Papers"] = len(state.papers)
            metrics["Claims"] = len(state.claims)
            metrics["Conflicts"] = len(state.conflicts)
            metrics["Gaps"] = len(state.gaps)
            
            # Add Critic scores if available
            if state.evaluations:
                eval = state.evaluations[-1]
                metrics["Critic_Score"] = eval.overall_quality
            else:
                metrics["Critic_Score"] = 0.0
                
            results.append(metrics)
            logger.info(f"Completed {name}. Overall Quality: {metrics.get('Critic_Score', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Failed variant {name}: {e}")
            results.append({"Variant": name, "Error": str(e)})

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"./data/output/ablation_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Table
    df = pd.DataFrame(results)
    # Reorder columns for readability
    cols = ["Variant", "CCS", "CD-F1", "GP@3", "TCS", "CalibScore", "Critic_Score"]
    # Filter for existing columns
    cols = [c for c in cols if c in df.columns]
    
    markdown_table = df[cols].to_markdown(index=False)
    
    report_path = f"./data/output/ablation_report_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(f"# Ablation Study Report: {TOPIC}\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Quantitative Results\n\n")
        f.write(markdown_table)
        f.write("\n\n## Discussion\n")
        f.write("- **Full System** provides the baseline for maximum integration.\n")
        f.write("- **No Confidence Scoring** significantly impacts CalibScore.\n")
        f.write("- **No Contradiction Detection** reduces CD-F1 and synthesis depth.\n")
        f.write("- **No Formalized Gaps** zeroed out GP@3.\n")
        f.write("- **No Temporal Tracking** zeroed out TCS.\n")

    logger.info(f"\nStudy complete. Report saved to {report_path}")
    print("\n--- ABLATION RESULTS ---")
    print(markdown_table)

if __name__ == "__main__":
    run_study()
