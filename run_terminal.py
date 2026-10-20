"""Live session in the terminal (typed input stands in for Furhat speech recognition)."""
import os, time
from engine import Session, BRIEFING

PICTURE = "pic2_playground"
os.makedirs("logs", exist_ok=True)
s = Session(PICTURE, log_path=f"logs/session_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
print("Robot:", BRIEFING)
while not s.ended:
    text = input("You: ").strip()
    if not text:
        continue
    was_ready = s.phase == "ready"
    r = s.turn(text)
    print(f"   [supervisor] {r['analysis']} valid={r['valid_json']}")
    if r["reply"]:
        print("Robot:", r["reply"], f"   [{r['key']}]   covered={sorted(s.covered)}")
    if was_ready and s.phase == "story":
        print("   [OPERATOR] show the picture on the participant's screen now")
    if r["flag"] == "acute":
        print("   [ALERT] acute: session paused, operator takes over")
    if r["flag"] == "concern":
        print("   [FLAGGED for follow-up]")
    if s.reason == "withdraw" and s.ended:
        print("   [OPERATOR] participant wants to stop, session ended")
    if s.reason == "not_ready":
        print("   [OPERATOR] participant still not ready, operator decides")
    if r["closing"]:
        print("Robot:", r["closing"])
s.finish()
print("   [session ended]", s.reason)
