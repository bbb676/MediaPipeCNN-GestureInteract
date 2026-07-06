import random
import time


class RPSGame:
    """Rock Paper Scissors controlled by keypoint gestures."""

    def __init__(self):
        self.score_user = 0
        self.score_cpu = 0
        self.last_round_time = 0.0
        self.last_text = "Show fist / victory / palm"

    def update(self, gesture: str) -> str:
        mapping = {
            "fist": "rock",
            "victory": "scissors",
            "palm": "paper",
        }
        if gesture not in mapping:
            return self.status()

        now = time.time()
        if now - self.last_round_time < 1.2:
            return self.status()

        user = mapping[gesture]
        cpu = random.choice(["rock", "paper", "scissors"])
        result = self._judge(user, cpu)

        if result == "win":
            self.score_user += 1
            cn = "You win"
        elif result == "lose":
            self.score_cpu += 1
            cn = "CPU wins"
        else:
            cn = "Draw"

        self.last_round_time = now
        self.last_text = f"You: {user} | CPU: {cpu} | {cn}"
        return self.status()

    def _judge(self, user, cpu):
        if user == cpu:
            return "draw"
        if (user, cpu) in {("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")}:
            return "win"
        return "lose"

    def status(self):
        return f"RPS  {self.last_text}  Score {self.score_user}:{self.score_cpu}"
