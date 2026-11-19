import pytest
from unittest.mock import patch
from server.core.agent_system import (
    TrendDiscoveryAgent, ResearchAgent, CreativeDirectorAgent,
    QualityAssuranceAgent, LearningAgent
)
from server.core.orchestrator import AutonomousStudioOrchestrator
from server.core.model_gateway import ModelGateway, BaseModelProvider

class MockProvider(BaseModelProvider):
    def __init__(self):
        super().__init__("MockProvider")

    def generate_text(self, prompt, **kwargs):
        p = prompt.lower()
        if "storyboard" in p or "creative director" in p or "scene-by-scene" in p:
            return '{"title": "The Unhackable Internet", "hook": "Everything you know about cybersecurity is about to change.", "target_duration_sec": 45, "estimated_cost_usd": 0.05, "scenes": [{"scene_number": 1, "duration_estimate_sec": 10.0, "speaker": "Host", "narration": "What if data could teleport?", "visual_description": "Glowing fiber optics with quantum nodes", "camera_framing": "Close-up"}]}'
        elif "quality assurance" in p or "evaluate this video script plan" in p:
            return '{"passed": true, "score": 95, "technical_defects": [], "creative_improvements": ["Pacing is solid."], "recommended_repairs": []}'
        elif "trend" in p or "opportunity" in p:
            return '[{"title": "The Quantum Internet Shift", "relevance_score": 95, "velocity_score": 90, "search_interest": 88, "competition_estimate": "Low", "opportunity_score": 91, "target_region": "Global", "why_it_works": "Emerging tech fascination"}]'
        elif "research the topic" in p or "authoritative research analyst" in p or "facts_ledger" in p:
            return '{"topic": "Quantum Internet", "summary": "Quantum networks enable unhackable transmission.", "key_insights": ["Entanglement enables secure keys", "Prototypes are active globally"], "facts_ledger": [{"claim": "Quantum key distribution works over fiber.", "source_reference": "Nature 2026", "confidence": 98, "is_factual": true}]}'
        return "Generic response"

@pytest.fixture
def mock_gateway():
    gateway_instance = ModelGateway()
    mock_p = MockProvider()
    gateway_instance.providers = {"mock": mock_p}
    gateway_instance.get_preferred_order = lambda task="general": ["mock"]
    return gateway_instance

def test_trend_discovery_agent(mock_gateway):
    with patch("server.core.agent_system.gateway", mock_gateway):
        agent = TrendDiscoveryAgent()
        opps = agent.discover_opportunities("quantum", region="United States")
        assert len(opps) >= 1
        assert opps[0].title == "The Quantum Internet Shift"
        assert opps[0].opportunity_score == 91

def test_research_agent(mock_gateway):
    with patch("server.core.agent_system.gateway", mock_gateway):
        agent = ResearchAgent()
        res = agent.research_topic("Quantum Internet")
        assert len(res.facts_ledger) >= 1
        assert res.facts_ledger[0].confidence == 98

def test_creative_director_agent(mock_gateway):
    with patch("server.core.agent_system.gateway", mock_gateway):
        agent = CreativeDirectorAgent()
        research = agent.research_topic = lambda t: None
        from server.core.agent_system import ResearchLedger
        mock_r = ResearchLedger(topic="Quantum", summary="Test summary", key_insights=[], facts_ledger=[])
        plan = agent.create_storyboard_plan("Quantum", mock_r, target_duration_sec=45)
        assert plan.title == "The Unhackable Internet"
        assert len(plan.scenes) >= 1

def test_learning_agent_memory():
    agent = LearningAgent()
    agent.record_preference("user_preference", "test_font", "Montserrat-Bold", confidence=100)
    val = agent.get_preference("user_preference", "test_font")
    assert val == "Montserrat-Bold"

def test_orchestrator_pipeline(mock_gateway):
    with patch("server.core.agent_system.gateway", mock_gateway):
        orchestrator = AutonomousStudioOrchestrator()
        result = orchestrator.produce_content_pipeline("Quantum Breakthroughs", duration_sec=45, auto_approve=True)
        assert result["title"] == "The Unhackable Internet"
        assert result["status"] == "ready_for_render"
        assert result["qa_passed"] is True

def test_urdu_and_roman_urdu_normalization():
    from pipeline import normalize_urdu_and_roman_urdu, sanitize_for_tts
    sample_text = "Aap kese hain? Yeh bohot acha hai aur mujhe dunya dekhni hai! 🚀"
    clean = sanitize_for_tts(sample_text)
    assert "kaisay" in clean
    assert "bohat" in clean
    assert "ach-cha" in clean
    assert "mujhey" in clean
    assert "duniya" in clean
    assert "🚀" not in clean

def test_multimodal_boost_calculation():
    from pipeline_shorts import get_multimodal_boost
    audio_energy = [0.1, 0.2, 0.9, 0.95, 0.8]
    visual_motion = [0.05, 0.1, 0.8, 0.85, 0.7]
    boost = get_multimodal_boost(2.0, 4.0, audio_energy, visual_motion, interval=1.0)
    assert boost > 5.0

def test_scene_regeneration_and_updates():
    from server.core.queue import create_job, save_stage_checkpoint
    from server.core.agent_system import ScriptScene
    job_id = create_job("test_scene_plan", job_type="plan")
    
    mock_plan = {
        "title": "AI Revolution",
        "scenes": [
            {"scene_number": 1, "duration_estimate_sec": 7.0, "speaker": "Host", "narration": "Intro", "visual_description": "Intro visual", "camera_framing": "Wide", "locked": False},
            {"scene_number": 2, "duration_estimate_sec": 8.0, "speaker": "Host", "narration": "Body", "visual_description": "Body visual", "camera_framing": "Close-up", "locked": True}
        ]
    }
    save_stage_checkpoint(job_id, "storyboard", mock_plan)
    
    # Test update_storyboard_scenes
    updated_scenes = mock_plan["scenes"]
    updated_scenes[0]["locked"] = True
    AutonomousStudioOrchestrator.update_storyboard_scenes(job_id, updated_scenes)
    
    from server.core.queue import get_stage_checkpoint
    saved = get_stage_checkpoint(job_id, "storyboard")
    assert saved["scenes"][0]["locked"] is True
