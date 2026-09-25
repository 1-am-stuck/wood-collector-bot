import os


def test_load_repo_env_sets_wandb_key(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    (tmp_path / ".env").write_text("WANDB_API_KEY=test-key-from-dotenv\n", encoding="utf-8")
    from load_env import load_repo_env, wandb_api_key

    path = load_repo_env(tmp_path)
    assert path == tmp_path / ".env"
    assert os.environ["WANDB_API_KEY"] == "test-key-from-dotenv"
    assert wandb_api_key(tmp_path) == "test-key-from-dotenv"
