from analysis_agent.domain.analysis import AnalysisReport, analyze_material


class DeterministicMaterialAnalyzer:
    def analyze(self, material: str) -> AnalysisReport:
        return analyze_material(material)
