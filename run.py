"""
CLI Entry Point for the Uncertainty-Aware Scientific Claim Synthesis system.
"""

import argparse
import sys
from loguru import logger

from src.config.settings import get_settings
from src.orchestration.autonomy_loop import AutonomyLoop


def main():
    parser = argparse.ArgumentParser(description="Uncertainty-Aware Scientific Claim Synthesis")
    parser.add_argument("topic", type=str, help="The research topic to synthesize")
    parser.add_argument("--max-papers", type=int, default=10, help="Maximum number of papers to retrieve")
    parser.add_argument("--iterations", type=int, default=2, help="Maximum autonomy iterations")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("./data/pipeline.log", level="DEBUG")
    
    logger.info(f"Initializing system for topic: '{args.topic}'")
    
    settings = get_settings()
    settings.max_papers = args.max_papers
    settings.max_autonomy_iterations = args.iterations
    
    # Check API keys
    if not settings.google_api_key and not settings.openai_api_key and not settings.anthropic_api_key:
        logger.error("No LLM API key configured in .env. Exiting.")
        sys.exit(1)
        
    loop = AutonomyLoop(settings)
    
    try:
        state = loop.execute(args.topic)
        
        logger.info("\n" + "="*50)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*50)
        logger.info(f"Papers analyzed: {len(state.papers)}")
        logger.info(f"Claims extracted: {len(state.claims)}")
        logger.info(f"Contradictions found: {len(state.conflicts)}")
        logger.info(f"Gaps identified: {len(state.gaps)}")
        
        if state.review:
            logger.info(f"Review sections generated: {len(state.review.sections)}")
            
            output_path = f"{settings.output_dir}/review_{args.topic.replace(' ', '_')[:20]}.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Literature Review: {state.topic}\n\n")
                for section in state.review.sections:
                    f.write(f"## {section.title}\n{section.content}\n\n")
                f.write(state.review.contested_claims_section + "\n\n")
                f.write(state.review.research_gaps_section + "\n\n")
                f.write(state.review.temporal_narrative + "\n\n")
                f.write(f"## Conclusion\n{state.review.conclusion}")
                
            logger.info(f"Review saved to {output_path}")
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")

if __name__ == "__main__":
    main()
