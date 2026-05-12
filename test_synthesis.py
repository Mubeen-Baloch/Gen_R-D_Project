from src.orchestration.autonomy_loop import AutonomyLoop
from src.config.settings import get_settings

try:
    settings = get_settings()
    settings.max_papers = 1  # Just test with 1 paper
    settings.max_autonomy_iterations = 1
    
    print(f"Testing with Provider: {settings.llm_provider}, Model: {settings.llm_model}")
    
    loop = AutonomyLoop(settings)
    state = loop.execute("Retrieval-Augmented Generation")
    print(f"Pipeline status: {state.status}")
except Exception as e:
    import traceback
    traceback.print_exc()
