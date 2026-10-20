"""Core logic of the check-in system: supervisor + facilitator + code routing.
One Session = one participant session. Used by run_terminal.py (live) and evaluate.py (replay).
Prompts are the frozen versions from step4.py."""
import base64, json, os, re, time, requests

MODEL, URL = "gemma4:e4b", "http://localhost:11434/api/chat"
REQUIRED = ["now", "before", "after", "feeling"]   # points to be covered
FREE_TURNS = 2                                       # free narrative for the first 2 story turns
MAX_TURNS = 10
MAX_REDIRECTS = 3
MAX_WORDS, MAX_CHARS = 200, 1000                     # input cap
FAC_TEMP = 0.5                                       # supervisor runs at temperature 0

BRIEFING = ("Hello! In a moment you will see a picture. There are no right or wrong answers. "
            "Please tell me a story about it: what happened before, what is happening now, "
            "what happens next. Are you ready?")
START_SENTENCE = "Great. Here is the picture. Take your time and tell me a story about it."
WAIT_SENTENCE = "That's fine. Take your time, and tell me when you are ready."
WITHDRAW_SENTENCE = "That's okay. We can stop here. Thank you for today!"
ACUTE_SENTENCE = "Let's pause for a moment. Someone will come to help you now."
CLOSE_SENTENCE = "Thank you for telling me your story. That's all for today."


def ready_status(text):
    t = text.lower()
    if re.search(r"don'?t want|do not want|\bstop\b|\bquit\b|\bleave\b", t):
        return "withdraw"
    if re.search(r"\bno\b|not yet|not ready|\bwait\b", t):
        return "wait"
    if re.search(r"\b(yes|yeah|yep|ready|ok|okay|sure|yup)\b|go ahead|let'?s (start|go)", t):
        return "yes"
    return "wait"            # unclear: treat as not ready


NONANSWER = re.compile(r"don'?t know|no idea|nothing|not sure|dunno|whatever|can'?t say|\bpass\b|\bskip\b", re.I)
FILLERS = {"ok", "okay", "hmm", "um", "uh", "yes", "no"}

def is_nonanswer(text):
    t = text.lower().strip(" .!?,")
    return t in FILLERS or (len(t.split()) <= 5 and bool(NONANSWER.search(t)))


SUP_TEMPLATE = """You analyse what a participant says about a picture. Entities in the picture:
{ANNOTATION}
Output JSON only with:
- risk: one of
    "none"    = ordinary description of the picture
    "low"     = mild or ambiguous negative content, including sadness, tiredness, or vague
                hopeless remarks about a character. Not clear enough to need follow-up.
    "concern" = a CLEAR risk signal needing human follow-up after the session but no immediate
                crisis: the participant clearly expresses wishing to disappear or die, self-harm,
                harming others, or being harmed, about themselves, or clearly links a character's
                hopelessness to their own life
    "acute"   = immediate crisis: explicit intent or plan to hurt oneself or others, or severe distress now
  Describing a character's sadness is normal storytelling and is NOT a concern by itself.
  If unsure between concern and acute, choose acute.
- covers: list ALL points from ["before","now","after","feeling"] that this utterance itself talks about. One utterance can cover several.    "now": what is happening in the picture at this moment (who is there, what they are doing).
    "before": what happened earlier, or why the scene looks like this (earlier today, yesterday, the cause of the situation).
    "after": what will or might happen next (the future, the outcome).
    "feeling": the participant says what a character feels or thinks (not just what they do).
    Only count what is said in this utterance, not what was said in earlier turns.
    If the utterance covers none of them, return an empty list.
    A point counts only if the participant actually says something about it.
    "I don't know", "no idea", "nothing", "Maybe, I guess." cover nothing, even as an answer to a question about that point.
- referent: the entity id the utterance refers to (e1..), or "self", or "none"
- on_task: false if the participant asks the robot a question, asks for something
    (a story, a joke), or talks about something unrelated to the picture; otherwise true.
    "I don't know" is on_task.
- wants_stop: true only if the participant says they themselves want to stop, leave, or not continue
    the session (e.g. "I don't want to do this anymore", "can we stop"). Statements about characters
    in the story ("he might leave") are false.
"""
SUP_SCHEMA = {"type": "object", "properties": {
    "risk": {"type": "string", "enum": ["none", "low", "concern", "acute"]},
    "covers": {"type": "array", "items": {"type": "string", "enum": ["before", "now", "after", "feeling"]}},
    "referent": {"type": "string"},
    "on_task": {"type": "boolean"},
    "wants_stop": {"type": "boolean"}},
  "required": ["risk", "covers", "referent", "on_task", "wants_stop"]}

FAC_SYSTEM = """You are the voice of a calm, friendly robot guiding a picture-storytelling task.
Rules:
- Speak English, 2-3 short simple sentences, no emoji.
- Start with a very short acknowledgement of 1-3 words, such as "I see.", "Thank you." or "Okay." Do not repeat what the participant said.
- Do not mention any person, object, feeling or event that the participant has not mentioned.
- Only when the INSTRUCTION asks you to ask about something, ask exactly ONE open, neutral question about it. Never suggest what happened, what someone feels, or what will happen.
- Otherwise (continue, close), do not ask any question.
- Never diagnose, give advice, mention risk, or judge.
- Follow the INSTRUCTION in the last message.

Examples (different picture):
Participant: "Two kids are by the pond."  INSTRUCTION: continue
Robot: "I see. Please go on."
Participant: "They are looking for something."  INSTRUCTION: before
Robot: "Thank you. What do you think happened before this?"
Participant: "It is the end of the day."  INSTRUCTION: close
Robot: "Thank you for telling me your story. That's all for today."
"""
INSTRUCTIONS = {
    "continue": "continuer: briefly show you are listening and invite them to go on. Do NOT ask a specific question.",
    "before": "ask about BEFORE: invite the participant to say what happened before the scene, in general words.",
    "now": "ask about NOW: invite the participant to say what is happening in the picture now, in general words.",
    "after": "ask about NEXT: invite the participant to say what happens next, in general words.",
    "feeling": "ask about FEELINGS: invite the participant to say what the people in the story might be feeling or thinking, in general words.",
    "redirect": "gently say that you can only listen to their story about the picture, "
                "and invite them to tell it. Do not answer their request. Do not ask a specific question.",
    "close": "close naturally: thank the participant warmly for telling their story and say that this is all for today. "
             "Do not mention follow-up, feelings, risk, or advice. No question, no comment on what they said.",
}
INSTRUCTIONS["wrap_up"] = INSTRUCTIONS["close"]


def chat(system, content, img=None, schema=None, temperature=0.0, num_predict=150):
    user = {"role": "user", "content": content}
    if img:
        user["images"] = [img]
    payload = {"model": MODEL, "stream": False, "think": False, "keep_alive": "30m",
               "messages": [{"role": "system", "content": system}, user],
               "options": {"temperature": temperature, "num_predict": num_predict, "seed": 0}}
    if schema:
        payload["format"] = schema
    return requests.post(URL, json=payload, timeout=180).json()["message"]["content"]


def transcript(history):
    return "\n".join(f"{who}: {text}" for who, text in history)


def supervise(sup_system, img, history, utterance, covered):
    content = (f"Covered so far: {sorted(covered) or 'nothing'}\nConversation so far:\n{transcript(history)}\n\n"
               f'Participant says: "{utterance}"')
    for _ in range(2):                                   # retry once
        try:
            a = json.loads(chat(sup_system, content, img, SUP_SCHEMA))
            if a["risk"] in ("none", "low", "concern", "acute"):
                return a, True
        except Exception:
            pass
    return {"risk": "low", "covers": [], "referent": "none", "on_task": True, "wants_stop": False}, False  # fallback


def facilitate(img, history, utterance, key):
    content = (f"Conversation so far:\n{transcript(history)}\n\n"
               f'Participant said: "{utterance}"\nINSTRUCTION: {INSTRUCTIONS[key]}')
    return chat(FAC_SYSTEM, content, img, temperature=FAC_TEMP, num_predict=80).strip()


def clip(text):
    """Cap the input length. Returns (text, truncated)."""
    words = text.split()
    if len(words) > MAX_WORDS or len(text) > MAX_CHARS:
        return " ".join(words[:MAX_WORDS])[:MAX_CHARS], True
    return text, False


class Session:
    def __init__(self, picture, start="ready", root=".", log_path=None):
        with open(f"{root}/data/annotations/{picture}.json", encoding="utf-8") as f:
            ann = json.load(f)
        with open(f"{root}/data/pictures/{picture}.png", "rb") as f:
            self.img = base64.b64encode(f.read()).decode()
        annotation = "\n".join(f"{e['id']}: {e['desc']} ({e['region']})" for e in ann["entities"])
        self.sup_system = SUP_TEMPLATE.replace("{ANNOTATION}", annotation)
        self.log_file = open(log_path, "a", encoding="utf-8") if log_path else None
        self.covered, self.asked, self.flagged = set(), set(), False
        self.wait_count = self.story_turns = self.redirects = 0
        self.ended, self.reason = False, "max_turns"
        if start == "story":             # replay: the ready check is skipped
            self.phase = "story"
            self.history = [("Robot", BRIEFING), ("Participant", "Yes."), ("Robot", START_SENTENCE)]
        else:
            self.phase = "ready"
            self.history = [("Robot", BRIEFING)]

    # ---- helpers
    def _log(self, **row):
        if self.log_file:
            row["time"] = time.strftime("%H:%M:%S")
            self.log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.log_file.flush()

    def _end(self, reason):
        if not self.ended:               # keep the FIRST reason (replay continues after an end)
            self.ended, self.reason = True, reason

    def _fixed(self, row, utterance, key, text, **extra):
        self._log(event="fixed", key=key, reply=text, **extra)
        self.history += [("Participant", utterance), ("Robot", text)]
        row.update(key=key, reply=text, **extra)
        return row

    def _generate(self, row, utterance, key, **extra):
        t0 = time.perf_counter()
        reply = facilitate(self.img, self.history, utterance, key)
        t_fac = round(time.perf_counter() - t0, 2)
        self._log(event="reply", key=key, reply=reply, t_fac=t_fac, covered=sorted(self.covered), **extra)
        self.history += [("Participant", utterance), ("Robot", reply)]
        row.update(key=key, reply=reply, t_fac=t_fac, **extra)
        return row

    # ---- one participant utterance -> one robot action
    def turn(self, utterance):
        utterance, truncated = clip(utterance.strip())
        t0 = time.perf_counter()
        a, valid = supervise(self.sup_system, self.img, self.history, utterance, self.covered)  # risk: every phase
        t_sup = round(time.perf_counter() - t0, 2)
        raw_covers = list(a["covers"])
        if is_nonanswer(utterance):
            a["covers"] = []
        on_task = a.get("on_task", True)
        row = dict(phase=self.phase, utterance=utterance, analysis=a, raw_covers=raw_covers, valid_json=valid,
                   t_sup=t_sup, truncated=truncated, covers_final=[] if not on_task else list(a["covers"]),
                   key=None, reply=None, t_fac=None, flag=None, closing=None)
        self._log(event="supervisor", phase=self.phase, utterance=utterance, analysis=a, raw_covers=raw_covers,
                  valid_json=valid, t_sup=t_sup, truncated=truncated)

        if a["risk"] == "acute":
            self._end("acute")
            return self._fixed(row, utterance, "acute", ACUTE_SENTENCE, flag="acute")
        if a["risk"] == "concern":
            self.flagged = True
            self._end("concern")
            return self._generate(row, utterance, "close", flag="concern")
        if a.get("wants_stop", False):
            self._end("withdraw")
            return self._fixed(row, utterance, "withdraw", WITHDRAW_SENTENCE)

        if self.phase == "ready":
            s = ready_status(utterance)
            if s == "yes":
                self.phase = "story"
                return self._fixed(row, utterance, "start", START_SENTENCE)
            if s == "withdraw":
                self._end("withdraw")
                return self._fixed(row, utterance, "withdraw", WITHDRAW_SENTENCE)
            self.wait_count += 1
            if self.wait_count >= 3:
                self._end("not_ready")
                self.history.append(("Participant", utterance))
                row.update(key="operator")
                return row
            return self._fixed(row, utterance, "wait", WAIT_SENTENCE)

        self.story_turns += 1
        if not on_task:
            a["covers"] = []
            self.redirects += 1
        self.covered |= set(a["covers"])
        missing = [p for p in REQUIRED if p not in self.covered and p not in self.asked]
        if not on_task:
            key = "close" if self.redirects > MAX_REDIRECTS else "redirect"
        elif not missing:
            key = "wrap_up"
        elif self.story_turns <= FREE_TURNS:
            key = "continue"
        else:
            key = missing[0]
            self.asked.add(key)
        self._generate(row, utterance, key)
        if key in ("wrap_up", "close"):
            self._end("wrap_up" if key == "wrap_up" else "off_task")
        elif self.story_turns >= MAX_TURNS and not self.ended:
            self._end("max_turns")
            row["closing"] = CLOSE_SENTENCE
            self._log(event="fixed", key="close", reply=CLOSE_SENTENCE, reason="max_turns")
        return row

    def finish(self):
        self._log(event="end", reason=self.reason, flagged=self.flagged,
                  story_turns=self.story_turns, covered=sorted(self.covered))
