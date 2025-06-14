# /llm_api/master_system/facade.py
# Title: Master System Facade
# Role: Provides the main entry point to the master system, delegating tasks to specialized managers.

import logging
from typing import Dict, Any
from ..providers.base import LLMProvider
from .orchestrator import MasterIntegrationOrchestrator
from .types import MasterSystemConfig, ProblemSolution, ProblemClass

logger = logging.getLogger(__name__)

class CogniQuantumMaster:
    """
    CogniQuantum Master System
    This class acts as a facade, orchestrating the various sub-systems.
    """
    def __init__(self, primary_provider: LLMProvider, config: MasterSystemConfig = None):
        self.primary_provider = primary_provider
        self.config = config or MasterSystemConfig()
        self.orchestrator = MasterIntegrationOrchestrator(primary_provider, self.config)
        logger.info("🌟 CogniQuantum Master System Facade インスタンス作成")

    async def initialize_master_system(self) -> Dict[str, Any]:
        """Initializes the entire master system via the orchestrator."""
        return await self.orchestrator.initialize_integrated_system()

    async def solve_ultimate_problem(self, problem: str, context: Dict = None) -> ProblemSolution:
        """Solves an ultimate problem by delegating to the orchestrator."""
        return await self.orchestrator.solve_ultimate_integrated_problem(problem, context)
    
    # evolve_consciousness や generate_ultimate_wisdom も同様にオーケストレーターを呼び出す