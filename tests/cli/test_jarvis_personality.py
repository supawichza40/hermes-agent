"""Tests for a JARVIS system prompt resolving through the personality
machinery the "Hey Jarvis" wake-word feature is documented to pair with.

Exercises ``HermesCLI._resolve_personality_prompt`` directly and the
``/personality`` command end-to-end with a ``jarvis`` entry under
``agent.personalities`` -- the config shape a user (or an installer
profile) adds to get a JARVIS-flavored system prompt. Mirrors the style of
``tests/cli/test_personality_none.py``'s ``TestPersonalityDictFormat``.
"""
from unittest.mock import MagicMock, patch

JARVIS_PERSONA = {
    "description": "JARVIS-style AI assistant",
    "system_prompt": (
        "You are JARVIS, a highly capable AI assistant. Address the user "
        "formally, anticipate their needs, and speak with dry wit."
    ),
    "tone": "formal, dry wit",
    "style": "concise, proactive",
}


class TestResolvePersonalityPromptJarvis:
    def test_resolves_jarvis_dict_persona(self):
        from cli import HermesCLI

        result = HermesCLI._resolve_personality_prompt(JARVIS_PERSONA)

        assert "JARVIS" in result
        assert "Tone: formal, dry wit" in result
        assert "Style: concise, proactive" in result

    def test_resolves_jarvis_string_persona(self):
        from cli import HermesCLI

        result = HermesCLI._resolve_personality_prompt("You are JARVIS.")

        assert result == "You are JARVIS."


class TestPersonalityCommandSelectsJarvis:
    def _make_cli(self, personalities):
        from cli import HermesCLI
        cli = HermesCLI.__new__(HermesCLI)
        cli.personalities = personalities
        cli.system_prompt = ""
        cli.agent = MagicMock()
        cli.console = MagicMock()
        return cli

    def test_personality_jarvis_sets_jarvis_system_prompt(self):
        cli = self._make_cli({"jarvis": JARVIS_PERSONA})
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality jarvis")

        assert "JARVIS" in cli.system_prompt
        assert cli.agent is None  # forced re-init so the new prompt takes effect

    def test_personality_jarvis_persists_to_config(self):
        cli = self._make_cli({"jarvis": JARVIS_PERSONA})
        with patch("cli.save_config_value", return_value=True) as mock_save:
            cli._handle_personality_command("/personality jarvis")

        mock_save.assert_called_once_with("agent.system_prompt", cli.system_prompt)

    def test_personality_jarvis_string_form_also_works(self):
        cli = self._make_cli({"jarvis": "You are JARVIS, Tony Stark's AI."})
        with patch("cli.save_config_value", return_value=True):
            cli._handle_personality_command("/personality jarvis")

        assert cli.system_prompt == "You are JARVIS, Tony Stark's AI."

    def test_jarvis_not_a_builtin_personality_by_default(self):
        """Documents a real gap: the wake-word bring-up message advertises
        'Say "Hey Jarvis"' (cli.py's _enable_voice_mode), but no built-in
        JARVIS persona ships in CLI_CONFIG's agent.personalities -- a user
        must add their own agent.personalities.jarvis entry (as the tests
        above do) to get a JARVIS-flavored system prompt alongside the wake
        word. This test intentionally fails if a future change adds one,
        so the change is a deliberate update to this expectation rather
        than a silent behavior change.
        """
        from cli import CLI_CONFIG

        builtin_personalities = CLI_CONFIG["agent"]["personalities"]
        assert "jarvis" not in builtin_personalities
